import torch
import os
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib as mpl
import cv2
import numpy as np
import matplotlib.cm as cm
import viser
import viser.transforms as tf
import time
import trimesh
import dataclasses
from scipy.spatial.transform import Rotation
from src.dust3r.viz import (
    add_scene_cam,
    CAM_COLORS,
    OPENGL,
    pts3d_to_trimesh,
    cat_meshes,
)


def todevice(batch, device, callback=None, non_blocking=False):
    """Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy)."""
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(todevice(x, device) for x in batch)

    x = batch
    if device == "numpy":
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x


to_device = todevice  # alias


def to_numpy(x):
    return todevice(x, "numpy")


@dataclasses.dataclass
class CameraState(object):
    fov: float
    aspect: float
    c2w: np.ndarray

    def get_K(self, img_wh):
        W, H = img_wh
        focal_length = H / 2.0 / np.tan(self.fov / 2.0)
        K = np.array(
            [
                [focal_length, 0.0, W / 2.0],
                [0.0, focal_length, H / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return K


class Enhanced3DPointCloudViewer:
    """Enhanced Point Cloud Viewer with 3D Bounding Box Support"""
    
    def __init__(
        self,
        model,
        state_args,
        pc_list,
        color_list,
        conf_list,
        cam_dict,
        image_mask=None,
        edge_color_list=None,
        device="cpu",
        port=8080,
        show_camera=True,
        vis_threshold=1,
        size=512,
        visualization_modes=None,
        bounding_boxes=None  # NEW: 3D bounding boxes
    ):
        self.model = model
        self.size = size
        self.state_args = state_args
        self.port = port
        self.server = viser.ViserServer(port=port)
        self.server.set_up_direction("-y")
        self.device = device
        self.conf_list = conf_list
        self.vis_threshold = vis_threshold
        self.tt = lambda x: torch.from_numpy(x).float().to(device)
        
        # Enhanced features
        self.visualization_modes = visualization_modes
        self.bounding_boxes = bounding_boxes  # List of lists of BoundingBox3D objects
        self.current_viz_mode = 'original'
        self.mask_mode_enabled = visualization_modes is not None
        self.bbox_mode_enabled = bounding_boxes is not None
        
        # Colors for different classes
        self.class_colors = {
            'zebra': [1.0, 0.6, 0.2],    # Bright orange
            'ground': [0.2, 1.0, 0.2],   # Bright green
            'sky': [0.3, 0.7, 1.0],      # Bright blue
            'person': [1.0, 0.2, 0.6],   # Bright pink
            'car': [0.8, 0.2, 1.0],      # Purple
            'building': [1.0, 1.0, 0.2], # Yellow
            'tree': [0.0, 0.8, 0.4],     # Forest green
            'rhino': [0.9, 0.5, 0.1],    # Orange-brown
            'rhinoceros': [0.9, 0.5, 0.1], # Same as rhino
        }
        
        self.pcs, self.all_steps = self.read_data(
            pc_list, color_list, conf_list, edge_color_list
        )
        self.cam_dict = cam_dict
        self.num_frames = len(self.all_steps)
        self.image_mask = image_mask
        self.show_camera = show_camera
        self.on_replay = False
        self.vis_pts_list = []
        self.traj_list = []
        self.orig_img_list = [x[0] for x in color_list]
        self.via_points = []

        # Setup enhanced GUI
        self._setup_enhanced_gui()
        
        self.pc_handles = []
        self.cam_handles = []
        self.bbox_handles = []  # NEW: For bounding box visualization
        self.label_handles = []  # NEW: For floating labels
        
        self.server.on_client_connect(self._connect_client)
    
    def _setup_enhanced_gui(self):
        """Setup enhanced GUI with mask and bounding box controls"""
        
        gui_reset_up = self.server.gui.add_button(
            "Reset up direction",
            hint="Set the camera control 'up' direction to the current camera's 'up'.",
        )

        @gui_reset_up.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            assert client is not None
            client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                [0.0, -1.0, 0.0]
            )

        button3 = self.server.gui.add_button("4D (Only Show Current Frame)")
        button4 = self.server.gui.add_button("3D (Show All Frames)")
        self.is_render = False
        self.fourd = False

        @button3.on_click
        def _(event: viser.GuiEvent) -> None:
            self.fourd = True

        @button4.on_click
        def _(event: viser.GuiEvent) -> None:
            self.fourd = False

        # Original controls
        self.focal_slider = self.server.add_gui_slider(
            "Focal Length",
            min=0.1,
            max=99999,
            step=1,
            initial_value=533,
        )

        self.psize_slider = self.server.add_gui_slider(
            "Point Size",
            min=0.0001,
            max=0.1,
            step=0.0001,
            initial_value=0.005,
        )
        self.camsize_slider = self.server.add_gui_slider(
            "Camera Size",
            min=0.01,
            max=0.5,
            step=0.01,
            initial_value=0.1,
        )
        
        # NEW: Mask visualization controls
        if self.mask_mode_enabled:
            with self.server.gui.add_folder("🎨 Mask Visualization", expand_by_default=True):
                self.viz_mode_text = self.server.gui.add_text(
                    "Current Mode", 
                    initial_value=f"🎯 {self.current_viz_mode.title()}"
                )
                
                # Visualization mode buttons
                self.original_button = self.server.gui.add_button("📷 Original Colors")
                self.overlay_button = self.server.gui.add_button("🎨 Mask Overlay") 
                self.highlight_button = self.server.gui.add_button("✨ Mask Highlight")
                self.mask_only_button = self.server.gui.add_button("🎯 Masks Only")
                
                # Blend strength slider
                self.blend_alpha_slider = self.server.add_gui_slider(
                    "Overlay Strength",
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    initial_value=0.6,
                )
                
                # Instance info
                if 'instance_labels' in self.visualization_modes:
                    total_instances = 0
                    total_masked_points = 0
                    for labels in self.visualization_modes['instance_labels']:
                        unique_labels = np.unique(labels)
                        total_instances += len(unique_labels[unique_labels > 0])
                        total_masked_points += np.sum(labels > 0)
                    
                    self.instance_info = self.server.gui.add_text(
                        "Instances Detected",
                        initial_value=f"🔍 {total_instances} instances, {total_masked_points:,} points"
                    )
                
                # Setup mask button callbacks
                self.original_button.on_click(lambda _: self._switch_visualization_mode('original'))
                self.overlay_button.on_click(lambda _: self._switch_visualization_mode('overlay')) 
                self.highlight_button.on_click(lambda _: self._switch_visualization_mode('highlight'))
                self.mask_only_button.on_click(lambda _: self._switch_visualization_mode('mask_only'))
                
                @self.blend_alpha_slider.on_update
                def _(_) -> None:
                    if self.current_viz_mode in ['overlay', 'highlight']:
                        self._refresh_visualization()

        # NEW: 3D Bounding Box controls
        if self.bbox_mode_enabled:
            with self.server.gui.add_folder("📦 3D Bounding Boxes", expand_by_default=True):
                # Count total bounding boxes
                total_bboxes = sum(len(frame_bboxes) for frame_bboxes in self.bounding_boxes)
                self.bbox_info = self.server.gui.add_text(
                    "Bounding Boxes",
                    initial_value=f"📦 {total_bboxes} 3D bounding boxes detected"
                )
                
                # Show/hide bounding boxes
                self.show_bboxes = self.server.gui.add_checkbox(
                    "Show Bounding Boxes", initial_value=True
                )
                
                # Show/hide class labels
                self.show_labels = self.server.gui.add_checkbox(
                    "Show Class Labels", initial_value=True
                )
                
                # Box line thickness
                self.bbox_thickness_slider = self.server.add_gui_slider(
                    "Box Line Thickness",
                    min=0.001,
                    max=0.02,
                    step=0.001,
                    initial_value=0.005,
                )
                
                # Box transparency
                self.bbox_opacity_slider = self.server.add_gui_slider(
                    "Box Opacity",
                    min=0.1,
                    max=1.0,
                    step=0.05,
                    initial_value=0.8,
                )
                
                # Class filtering (checkboxes for each detected class)
                detected_classes = set()
                for frame_bboxes in self.bounding_boxes:
                    for bbox in frame_bboxes:
                        detected_classes.add(bbox.class_name)
                
                if detected_classes:
                    with self.server.gui.add_folder("🏷️ Filter by Class", expand_by_default=False):
                        self.class_checkboxes = {}
                        for class_name in sorted(detected_classes):
                            self.class_checkboxes[class_name] = self.server.gui.add_checkbox(
                                f"Show {class_name.title()}", initial_value=True
                            )
                
                # Callbacks for bounding box controls
                @self.show_bboxes.on_update
                def _(_) -> None:
                    self._refresh_bboxes()
                
                @self.show_labels.on_update
                def _(_) -> None:
                    self._refresh_bboxes()
                
                @self.bbox_thickness_slider.on_update
                def _(_) -> None:
                    self._refresh_bboxes()
                
                @self.bbox_opacity_slider.on_update
                def _(_) -> None:
                    self._refresh_bboxes()
                
                # Setup class filter callbacks
                if hasattr(self, 'class_checkboxes'):
                    for checkbox in self.class_checkboxes.values():
                        checkbox.on_update(lambda _: self._refresh_bboxes())

        @self.psize_slider.on_update
        def _(_) -> None:
            for handle in self.pc_handles:
                handle.point_size = self.psize_slider.value

        @self.camsize_slider.on_update
        def _(_) -> None:
            for handle in self.cam_handles:
                handle.scale = self.camsize_slider.value
                handle.line_thickness = 0.03 * handle.scale
    
    def _switch_visualization_mode(self, mode):
        """Switch between different visualization modes"""
        print(f"🎨 Switching to {mode} visualization mode...")
        
        self.current_viz_mode = mode
        self.viz_mode_text.value = f"🎯 {mode.title().replace('_', ' ')}"
        
        # Update all point clouds with new colors
        self._refresh_visualization()
        
        print(f"✅ Switched to {mode} mode")
    
    def _refresh_visualization(self):
        """Refresh the visualization with current mode"""
        if not self.mask_mode_enabled:
            return

        # Clear existing point cloud handles (safely handle already-removed nodes)
        for handle in self.pc_handles:
            try:
                handle.remove()
            except (KeyError, RuntimeError):
                pass  # Node was already removed
        self.pc_handles.clear()
        
        # Re-add point clouds with current visualization mode
        for i, step in enumerate(self.all_steps):
            if hasattr(self, 'frame_nodes') and len(self.frame_nodes) > i:
                # Only update if frames are initialized
                self._add_pc_with_mode(step, current_mode=self.current_viz_mode)
    
    def _refresh_bboxes(self):
        """Refresh bounding box visualization"""
        if not self.bbox_mode_enabled:
            return

        # Clear existing bbox handles (safely handle already-removed nodes)
        for handle in self.bbox_handles:
            try:
                handle.remove()
            except (KeyError, RuntimeError):
                pass  # Node was already removed
        for handle in self.label_handles:
            try:
                handle.remove()
            except (KeyError, RuntimeError):
                pass  # Node was already removed
        self.bbox_handles.clear()
        self.label_handles.clear()
        
        # Re-add bounding boxes for current frame
        if hasattr(self, 'frame_nodes'):
            for i, step in enumerate(self.all_steps):
                if len(self.frame_nodes) > i and self.frame_nodes[i].visible:
                    self._add_bboxes_for_frame(step)
    
    def _get_colors_for_mode(self, step, mode):
        """Get colors for a specific visualization mode"""
        if not self.mask_mode_enabled:
            return self.pcs[step]["color"]
        
        frame_idx = step  # Assuming step corresponds to frame index
        
        if mode == 'original':
            return self.visualization_modes['original_colors'][frame_idx]
        elif mode == 'overlay':
            return self.visualization_modes['overlay_colors'][frame_idx]
        elif mode == 'highlight':
            return self.visualization_modes['highlight_colors'][frame_idx]
        elif mode == 'mask_only':
            return self.visualization_modes['mask_only_colors'][frame_idx]
        else:
            return self.pcs[step]["color"]  # Fallback
    
    def _add_pc_with_mode(self, step, current_mode=None):
        """Add point cloud with specified visualization mode"""
        if current_mode is None:
            current_mode = self.current_viz_mode
            
        pc = self.pcs[step]["pc"]
        
        # Get colors based on current mode
        if self.mask_mode_enabled:
            color = self._get_colors_for_mode(step, current_mode)
        else:
            color = self.pcs[step]["color"]
            
        conf = self.pcs[step]["conf"]
        edge_color = self.pcs[step].get("edge_color", None)

        pred_pts, color = self.parse_pc_data(
            pc, color, conf, edge_color, set_border_color=True
        )

        self.vis_pts_list.append(pred_pts)
        self.pc_handles.append(
            self.server.add_point_cloud(
                name=f"/frames/{step}/pred_pts",
                points=pred_pts,
                colors=color,
                point_size=self.psize_slider.value,
            )
        )
    
    def _get_class_color(self, class_name):
        """Get color for a class"""
        if class_name.lower() in self.class_colors:
            return self.class_colors[class_name.lower()]
        else:
            # Generate color based on hash
            hash_val = hash(class_name.lower()) % 1000
            return [
                (hash_val * 0.618) % 1.0,
                ((hash_val * 0.618) * 2) % 1.0,
                ((hash_val * 0.618) * 3) % 1.0
            ]
    
    def _add_bboxes_for_frame(self, step):
        """Add 3D bounding boxes for a specific frame"""
        if not self.bbox_mode_enabled or not self.show_bboxes.value:
            return
            
        frame_idx = step
        if frame_idx >= len(self.bounding_boxes):
            return
            
        frame_bboxes = self.bounding_boxes[frame_idx]
        
        for bbox_idx, bbox in enumerate(frame_bboxes):
            # Check if this class should be shown
            if hasattr(self, 'class_checkboxes'):
                if bbox.class_name in self.class_checkboxes:
                    if not self.class_checkboxes[bbox.class_name].value:
                        continue
            
            # Get box corners and edges
            corners, edges = bbox.get_wireframe_edges()
            
            # Get class color
            class_color = self._get_class_color(bbox.class_name)
            
            # Add wireframe lines
            for edge in edges:
                start_point = corners[edge[0]]
                end_point = corners[edge[1]]
                
                # Create line geometry
                line_points = np.array([start_point, end_point])
                
                self.bbox_handles.append(
                    self.server.add_spline_catmull_rom(
                        name=f"/frames/{step}/bbox_{bbox_idx}_edge_{len(self.bbox_handles)}",
                        positions=line_points,
                        color=class_color,
                        line_width=self.bbox_thickness_slider.value * 1000,  # Scale for visibility
                        segments=2
                    )
                )
            
            # Add floating label
            if self.show_labels.value:
                label_text = f"{bbox.class_name.title()}\n{bbox.confidence:.2f}"
                label_position = bbox.center + np.array([0, 0, bbox.dimensions[2]/2 + 0.1])  # Above the box
                
                self.label_handles.append(
                    self.server.add_point_cloud(
                        name=f"/frames/{step}/label_{bbox_idx}",
                        points=label_position.reshape(1, 3),
                        colors=np.array(class_color).reshape(1, 3),
                        point_size=0.01,  # Very small, just for the label
                    )
                )
                
                # Add text label (if viser supports it)
                try:
                    self.label_handles.append(
                        self.server.add_text(
                            name=f"/frames/{step}/text_{bbox_idx}",
                            text=label_text,
                            position=label_position,
                            color=class_color
                        )
                    )
                except:
                    pass  # Fallback if text labels not supported

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        camera = client.camera
        c2w = np.concatenate(
            [
                np.concatenate(
                    [tf.SO3(camera.wxyz).as_matrix(), camera.position[:, None]], 1
                ),
                [[0, 0, 0, 1]],
            ],
            0,
        )
        return CameraState(
            fov=camera.fov,
            aspect=camera.aspect,
            c2w=c2w,
        )

    @staticmethod
    def generate_pseudo_intrinsics(h, w):
        focal = (h**2 + w**2) ** 0.5
        return np.array([[focal, 0, w // 2], [0, focal, h // 2], [0, 0, 1]]).astype(
            np.float32
        )

    def get_ray_map(self, c2w, h, w, intrinsics=None):
        if intrinsics is None:
            intrinsics = self.generate_pseudo_intrinsics(h, w)
        i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
        grid = np.stack([i, j, np.ones_like(i)], axis=-1)
        ro = c2w[:3, 3]
        rd = np.linalg.inv(intrinsics) @ grid.reshape(-1, 3).T
        rd = (c2w @ np.vstack([rd, np.ones_like(rd[0])])).T[:, :3].reshape(h, w, 3)
        rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
        ro = np.broadcast_to(ro, (h, w, 3))
        ray_map = np.concatenate([ro, rd], axis=-1)
        return ray_map

    def _connect_client(self, client: viser.ClientHandle):
        from src.dust3r.inference import inference_step
        from src.dust3r.utils.geometry import geotrf

        wxyz_panel = client.gui.add_text("wxyz:", f"{client.camera.wxyz}")
        position_panel = client.gui.add_text("position:", f"{client.camera.position}")
        fov_panel = client.gui.add_text(
            "fov:", f"{2 * np.arctan(self.size/self.focal_slider.value) * 180 / np.pi}"
        )
        aspect_panel = client.gui.add_text("aspect:", "1.0")

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            with self.server.atomic():
                wxyz_panel.value = f"{client.camera.wxyz}"
                position_panel.value = f"{client.camera.position}"
                fov_panel.value = (
                    f"{2 * np.arctan(self.size/self.focal_slider.value) * 180 / np.pi}"
                )
                aspect_panel.value = "1.0"

        gui_set_current_camera = client.gui.add_button(
            "Set Current Camera to Infer Raymap"
        )

        @gui_set_current_camera.on_click
        def _(_) -> None:
            try:
                cam = self.get_camera_state(client)
                cam.fov = 2 * np.arctan(self.size / self.focal_slider.value)
                cam.aspect = (512 / 384) if self.size==512 else 1.0
                pose = cam.c2w
                if self.size == 512:
                    intrins = self.generate_pseudo_intrinsics(384, 512)
                    raymap = torch.from_numpy(self.get_ray_map(pose, 384, 512, intrins))[
                        None
                    ].float()
                else:
                    intrins = self.generate_pseudo_intrinsics(224, 224)
                    raymap = torch.from_numpy(self.get_ray_map(pose, 224, 224, intrins))[
                        None
                    ].float()
                
                view = {
                    "img": torch.full((1, 3, 384, 512), torch.nan) if self.size==512 else torch.full((1, 3, 224, 224), torch.nan),
                    "ray_map": raymap,
                    "true_shape": torch.from_numpy(np.int32([raymap.shape[1:-1]])),
                    "idx": self.num_frames + 1,
                    "instance": str(self.num_frames + 1),
                    "camera_pose": torch.from_numpy(np.eye(4).astype(np.float32)).unsqueeze(
                        0
                    ),
                    "img_mask": torch.tensor(False).unsqueeze(0),
                    "ray_mask": torch.tensor(True).unsqueeze(0),
                    "update": torch.tensor(False).unsqueeze(0),
                    "reset": torch.tensor(False).unsqueeze(0),
                }
                print("Start Inference Raymap")
                output = inference_step(
                    view, self.state_args[-1], self.model, device=self.device
                )
                print("Finish Inference Raymap")
                pts3ds = output["pred"]["pts3d_in_self_view"].cpu().numpy()
                pts3ds = geotrf(pose[None], pts3ds)
                colors = 0.5 * (output["pred"]["rgb"].cpu().numpy() + 1.0)
                depthmap = output["pred"]["pts3d_in_self_view"].cpu().numpy()[0][..., -1]
                conf = output["pred"]["conf"].cpu().numpy()
                disp = 1.0 / depthmap
                pts3ds, colors = self.parse_pc_data(pts3ds, colors, set_border_color=True)
                mask = (conf > 1.0).reshape(-1)
                self.num_frames += 1
                self.pc_handles.append(
                    self.server.add_point_cloud(
                        name=f"/frames/{self.num_frames-1}/pred_pts",
                        points=pts3ds[mask],
                        colors=colors[mask],
                        point_size=0.005,
                    )
                )

                self.server.add_camera_frustum(
                    name=f"/frames/{self.num_frames-1}/camera",
                    fov=cam.fov,
                    aspect=cam.aspect,
                    wxyz=client.camera.wxyz,
                    position=client.camera.position,
                    scale=0.1,
                    color=[64, 179, 230],
                )
                print("Adding new pointcloud: ", pts3ds.shape)
            except Exception as e:
                print(e)

    @staticmethod
    def set_color_border(image, border_width=5, color=[1, 0, 0]):
        image[:border_width, :, 0] = color[0]  # Red channel
        image[:border_width, :, 1] = color[1]  # Green channel
        image[:border_width, :, 2] = color[2]  # Blue channel
        image[-border_width:, :, 0] = color[0]
        image[-border_width:, :, 1] = color[1]
        image[-border_width:, :, 2] = color[2]

        image[:, :border_width, 0] = color[0]
        image[:, :border_width, 1] = color[1]
        image[:, :border_width, 2] = color[2]
        image[:, -border_width:, 0] = color[0]
        image[:, -border_width:, 1] = color[1]
        image[:, -border_width:, 2] = color[2]

        return image

    def read_data(self, pc_list, color_list, conf_list, edge_color_list=None):
        pcs = {}
        step_list = []
        for i, pc in enumerate(pc_list):
            step = i
            pcs.update(
                {
                    step: {
                        "pc": pc,
                        "color": color_list[i],
                        "conf": conf_list[i],
                        "edge_color": (
                            None if edge_color_list[i] is None else edge_color_list[i]
                        ),
                    }
                }
            )
            step_list.append(step)
        normalized_indices = (
            np.array(list(range(len(pc_list))))
            / np.array(list(range(len(pc_list)))).max()
        )
        cmap = cm.viridis
        self.camera_colors = cmap(normalized_indices)
        return pcs, step_list

    def parse_pc_data(
        self,
        pc,
        color,
        conf=None,
        edge_color=[0.251, 0.702, 0.902],
        set_border_color=False,
    ):
        pred_pts = pc.reshape(-1, 3)  # [N, 3]

        if set_border_color and edge_color is not None:
            color = self.set_color_border(color[0], color=edge_color)
        if np.isnan(color).any():
            color = np.zeros((pred_pts.shape[0], 3))
            color[:, 2] = 1
        else:
            color = color.reshape(-1, 3)
        if conf is not None:
            conf = conf[0].reshape(-1)
            pred_pts = pred_pts[conf > self.vis_threshold]
            color = color[conf > self.vis_threshold]
        return pred_pts, color

    def add_pc(self, step):
        """Enhanced add_pc method with mask support"""
        self._add_pc_with_mode(step)

    def add_camera(self, step):
        cam = self.cam_dict
        focal = cam["focal"][step]
        pp = cam["pp"][step]
        R = cam["R"][step]
        t = cam["t"][step]

        q = tf.SO3.from_matrix(R).wxyz
        fov = 2 * np.arctan(pp[0] / focal)
        aspect = pp[0] / pp[1]
        self.traj_list.append((q, t))
        self.cam_handles.append(
            self.server.add_camera_frustum(
                name=f"/frames/{step}/camera",
                fov=fov,
                aspect=aspect,
                wxyz=q,
                position=t,
                scale=0.1,
                color=(50, 205, 50),
            )
        )

    def animate(self):
        with self.server.add_gui_folder("Playback"):
            gui_timestep = self.server.add_gui_slider(
                "Train Step",
                min=0,
                max=self.num_frames - 1,
                step=1,
                initial_value=0,
                disabled=False,
            )
            gui_next_frame = self.server.add_gui_button("Next Step", disabled=False)
            gui_prev_frame = self.server.add_gui_button("Prev Step", disabled=False)
            gui_playing = self.server.add_gui_checkbox("Playing", False)
            gui_framerate = self.server.add_gui_slider(
                "FPS", min=1, max=60, step=0.1, initial_value=1
            )
            gui_framerate_options = self.server.add_gui_button_group(
                "FPS options", ("10", "20", "30", "60")
            )

        @gui_next_frame.on_click
        def _(_) -> None:
            gui_timestep.value = (gui_timestep.value + 1) % self.num_frames

        @gui_prev_frame.on_click
        def _(_) -> None:
            gui_timestep.value = (gui_timestep.value - 1) % self.num_frames

        @gui_playing.on_update
        def _(_) -> None:
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

        @gui_framerate_options.on_click
        def _(_) -> None:
            gui_framerate.value = int(gui_framerate_options.value)

        prev_timestep = gui_timestep.value

        @gui_timestep.on_update
        def _(_) -> None:
            nonlocal prev_timestep
            current_timestep = gui_timestep.value
            with self.server.atomic():
                self.frame_nodes[current_timestep].visible = True
                self.frame_nodes[prev_timestep].visible = False
                
                # 🔥 CRITICAL FIX: Update bounding boxes for current frame
                if self.bbox_mode_enabled:
                    print(f"🔄 Switching to frame {current_timestep}, clearing old bboxes...")
                    
                    # Clear all existing bounding boxes
                    for handle in self.bbox_handles:
                        try:
                            handle.remove()
                        except:
                            pass
                    for handle in self.label_handles:
                        try:
                            handle.remove()
                        except:
                            pass
                    self.bbox_handles.clear()
                    self.label_handles.clear()
                    
                    # Add bounding boxes ONLY for current frame
                    print(f"📦 Adding bboxes for frame {current_timestep}")
                    self._add_bboxes_for_frame(current_timestep)
                    
            prev_timestep = current_timestep
            self.server.flush()

        self.server.add_frame(
            "/frames",
            show_axes=False,
        )
        self.frame_nodes = []
        for i in range(self.num_frames):
            step = self.all_steps[i]
            self.frame_nodes.append(
                self.server.add_frame(
                    f"/frames/{step}",
                    show_axes=False,
                )
            )
            self.add_pc(step)
            if self.show_camera:
                self.add_camera(step)
            # Add bounding boxes for first frame initially
            if i == 0 and self.bbox_mode_enabled:
                self._add_bboxes_for_frame(step)

        prev_timestep = gui_timestep.value
        
        # Enhanced status message
        print("\n" + "="*60)
        print("🚀 ENHANCED POINT CLOUD VIEWER WITH 3D BOUNDING BOXES")
        print("="*60)
        print(f"🌐 View at: http://localhost:{self.port}")
        if self.mask_mode_enabled:
            print(f"🎨 Mask visualization modes available!")
        if self.bbox_mode_enabled:
            total_bboxes = sum(len(frame_bboxes) for frame_bboxes in self.bounding_boxes)
            print(f"📦 {total_bboxes} 3D bounding boxes loaded!")
        print(f"📊 Use sliders to adjust point size and visibility")
        print(f"🎮 Use playback controls for animation")
        print("="*60)
        
        while True:
            if self.on_replay:
                pass
            else:
                if gui_playing.value:
                    gui_timestep.value = (gui_timestep.value + 1) % self.num_frames

                for i, frame_node in enumerate(self.frame_nodes):
                    frame_node.visible = (
                        i <= gui_timestep.value
                        if not self.fourd
                        else i == gui_timestep.value
                    )

            time.sleep(1.0 / gui_framerate.value)

    def run(self):
        self.animate()
        while True:
            time.sleep(10.0)


# Create an alias for backward compatibility
PointCloudViewer = Enhanced3DPointCloudViewer