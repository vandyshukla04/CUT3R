#!/usr/bin/env python3
"""
Simple 3D Bbox Editor - Fixed version without callback loops

Usage:
    python bbox_editor_v2.py --auto_bboxes results/DIR/bounding_boxes --output corrected_bboxes
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import viser

# Simple bbox class
class BBox3D:
    def __init__(self, center, dimensions, rotation_matrix, class_name, track_id, frame_idx):
        self.center = np.array(center)
        self.dimensions = np.array(dimensions)
        self.rotation_matrix = np.array(rotation_matrix)
        self.class_name = class_name
        self.track_id = track_id
        self.frame_idx = frame_idx

    def get_corners(self):
        """Get 8 corners"""
        l, w, h = self.dimensions
        corners_local = np.array([
            [-l/2, -w/2, -h/2], [l/2, -w/2, -h/2],
            [l/2, w/2, -h/2], [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2], [l/2, -w/2, h/2],
            [l/2, w/2, h/2], [-l/2, w/2, h/2]
        ])
        return (self.rotation_matrix @ corners_local.T).T + self.center

    def get_edges(self):
        """Get edge indices for wireframe"""
        return [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical
        ]


class SimpleBBoxEditor:
    """Minimal bbox editor - no callback loops!"""

    def __init__(self, auto_bboxes_dir, output_dir, point_clouds_dir=None, images_dir=None, port=8080):
        self.auto_bboxes_dir = Path(auto_bboxes_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store results dir for saving corrected bboxes
        self.results_dir = self.auto_bboxes_dir.parent

        # Try to auto-find point clouds
        if point_clouds_dir is None:
            pc_world = self.results_dir / "point_clouds_world"
            pc_local = self.results_dir / "point_clouds"
            if pc_world.exists():
                self.point_clouds_dir = pc_world
            elif pc_local.exists():
                self.point_clouds_dir = pc_local
            else:
                self.point_clouds_dir = None
        else:
            self.point_clouds_dir = Path(point_clouds_dir)

        # Images directory for 2D visualization
        self.images_dir = Path(images_dir) if images_dir else None

        # Load bbox data first (needed for frame_indices)
        self.auto_bboxes = self._load_bboxes()
        self.frame_indices = sorted(self.auto_bboxes.keys())
        self.corrections = {}  # track_id -> {frame_idx -> bbox}

        # Try to load camera parameters (needs frame_indices)
        self.cam_dict = self._load_camera_params()

        # Point clouds (loaded on demand)
        self.point_clouds = {}  # frame_idx -> (points, colors)

        # Viser server (use same pattern as viser_utils.py)
        self.server = viser.ViserServer(port=port)
        self.server.set_up_direction("-y")  # CRITICAL!

        # State
        self.current_frame_idx = 0
        self.current_frame = self.frame_indices[0] if self.frame_indices else 0
        self.selected_track = None

        # Scene handles - organized by type
        self.pc_handle = None
        self.bbox_handles = {}  # bbox_id -> list of handles
        self.gizmo_handle = None

        # Flag to prevent update loops
        self.updating_sliders = False

        # Setup UI
        self._setup_ui()

        # Initial render
        self._render_frame()

        print(f"\n{'='*60}")
        print(f"BBOX EDITOR READY")
        print(f"{'='*60}")
        print(f"Open: http://localhost:{port}")
        print(f"Frames: {len(self.frame_indices)}")
        print(f"{'='*60}\n")

    def _load_camera_params(self):
        """Load camera parameters from results dir"""
        try:
            # Check for single camera_parameters.npz file first
            cam_file = self.results_dir / "camera_parameters.npz"
            if cam_file.exists():
                data = np.load(cam_file)
                print(f"✓ Loaded camera parameters from {cam_file}")
                return {
                    'focal': data['focals'],
                    'pp': data['principal_points'],
                    'R': data['Rs'],
                    't': data['ts']
                }

            # Otherwise, load from individual camera/*.npz files (demo_masks.py format)
            camera_dir = self.results_dir / "camera"
            if camera_dir.exists():
                print(f"✓ Loading camera parameters from {camera_dir}")

                # Load camera params for each frame
                focals = []
                pps = []
                Rs = []
                ts = []

                for frame_idx in self.frame_indices:
                    cam_file = camera_dir / f"{frame_idx}.npz"
                    if not cam_file.exists():
                        # Try with zero-padded name
                        cam_file = camera_dir / f"{frame_idx:06d}.npz"

                    if cam_file.exists():
                        data = np.load(cam_file)
                        # Extract from pose and intrinsics
                        intrinsics = data['intrinsics']  # [3, 3]
                        pose = data['pose']  # [4, 4] c2w matrix

                        # Extract focal and principal point from intrinsics
                        focal = intrinsics[0, 0]  # fx (assuming square pixels)
                        pp = np.array([intrinsics[0, 2], intrinsics[1, 2]])

                        # Extract R and t from pose (c2w)
                        R = pose[:3, :3]
                        t = pose[:3, 3]

                        focals.append(focal)
                        pps.append(pp)
                        Rs.append(R)
                        ts.append(t)
                    else:
                        print(f"⚠️ Missing camera file for frame {frame_idx}")
                        return None

                if len(focals) == len(self.frame_indices):
                    return {
                        'focal': np.array(focals),
                        'pp': np.array(pps),
                        'R': np.array(Rs),
                        't': np.array(ts)
                    }
                else:
                    print(f"⚠️ Incomplete camera data: {len(focals)}/{len(self.frame_indices)} frames")
                    return None

        except Exception as e:
            print(f"⚠️ Could not load camera parameters: {e}")
            import traceback
            traceback.print_exc()

        return None

    def _load_bboxes(self):
        """Load auto bboxes from JSON files"""
        bboxes = {}
        for json_file in sorted(self.auto_bboxes_dir.glob("*.json")):
            frame_idx = int(json_file.stem)
            with open(json_file) as f:
                data = json.load(f)

            frame_bboxes = []
            for bbox_dict in data:
                bbox = BBox3D(
                    center=bbox_dict['center'],
                    dimensions=bbox_dict['dimensions'],
                    rotation_matrix=bbox_dict['rotation_matrix'],
                    class_name=bbox_dict['class_name'],
                    track_id=bbox_dict['track_id'],
                    frame_idx=frame_idx
                )
                frame_bboxes.append(bbox)

            bboxes[frame_idx] = frame_bboxes

        print(f"Loaded {len(bboxes)} frames")
        return bboxes

    def _load_point_cloud(self, frame_idx):
        """Load point cloud for a frame (PLY or NPZ format)"""
        if self.point_clouds_dir is None:
            return None, None

        # Check if already loaded
        if frame_idx in self.point_clouds:
            return self.point_clouds[frame_idx]

        # Try PLY first
        ply_file = self.point_clouds_dir / f"{frame_idx}.ply"
        if ply_file.exists():
            try:
                import trimesh
                mesh = trimesh.load(str(ply_file))
                points = np.array(mesh.vertices)

                # Get colors if available
                if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                    colors = np.array(mesh.visual.vertex_colors[:, :3]) / 255.0
                else:
                    colors = np.ones_like(points) * 0.5

                # Subsample if too large
                if len(points) > 50000:
                    indices = np.random.choice(len(points), 50000, replace=False)
                    points = points[indices]
                    colors = colors[indices]

                self.point_clouds[frame_idx] = (points, colors)
                return points, colors

            except Exception as e:
                print(f"Failed to load PLY {frame_idx}: {e}")

        return None, None

    def _setup_ui(self):
        """Setup GUI controls"""

        # Frame navigation buttons
        prev_frame_btn = self.server.add_gui_button("◀ Previous Frame")

        @prev_frame_btn.on_click
        def _(_):
            if self.current_frame_idx > 0:
                self.current_frame_idx -= 1
                self.current_frame = self.frame_indices[self.current_frame_idx]
                self.updating_sliders = True
                self.frame_slider.value = self.current_frame_idx
                self.updating_sliders = False
                self._render_frame()

        next_frame_btn = self.server.add_gui_button("Next Frame ▶")

        @next_frame_btn.on_click
        def _(_):
            if self.current_frame_idx < len(self.frame_indices) - 1:
                self.current_frame_idx += 1
                self.current_frame = self.frame_indices[self.current_frame_idx]
                self.updating_sliders = True
                self.frame_slider.value = self.current_frame_idx
                self.updating_sliders = False
                self._render_frame()

        # Frame slider
        self.frame_slider = self.server.add_gui_slider(
            "Frame",
            min=0,
            max=len(self.frame_indices) - 1,
            step=1,
            initial_value=0
        )

        @self.frame_slider.on_update
        def _(_):
            if not self.updating_sliders:
                self.current_frame_idx = int(self.frame_slider.value)
                self.current_frame = self.frame_indices[self.current_frame_idx]
                self._render_frame()

        # Track dropdown
        all_tracks = set()
        for bboxes in self.auto_bboxes.values():
            for bbox in bboxes:
                all_tracks.add(bbox.track_id)

        track_options = ["(None)"] + [f"Track {tid}" for tid in sorted(all_tracks)]
        self.track_dropdown = self.server.add_gui_dropdown(
            "Select Track",
            options=track_options
        )

        @self.track_dropdown.on_update
        def _(_):
            if self.track_dropdown.value == "(None)":
                self.selected_track = None
            else:
                self.selected_track = int(self.track_dropdown.value.split()[-1])
            self._update_selection()

        # Info display
        self.info_text = self.server.add_gui_text(
            "Info",
            initial_value="Select a track to edit",
            disabled=True
        )

        # Point cloud controls
        if self.point_clouds_dir:
            self.point_size_slider = self.server.add_gui_slider(
                "Point Size",
                min=0.001,
                max=0.02,
                step=0.001,
                initial_value=0.005
            )

            @self.point_size_slider.on_update
            def _(_):
                if self.pc_handle is not None:
                    self.pc_handle.point_size = self.point_size_slider.value

        # Dimension editing sliders (for selected bbox)
        # Use dim[0], dim[1], dim[2] labels so user can see which is which when dragging
        self.dim_0_slider = self.server.add_gui_slider(
            "dim[0]",
            min=0.1,
            max=5.0,
            step=0.01,
            initial_value=1.0
        )

        self.dim_1_slider = self.server.add_gui_slider(
            "dim[1]",
            min=0.1,
            max=5.0,
            step=0.01,
            initial_value=1.0
        )

        self.dim_2_slider = self.server.add_gui_slider(
            "dim[2]",
            min=0.1,
            max=5.0,
            step=0.01,
            initial_value=1.0
        )

        @self.dim_0_slider.on_update
        def _(_):
            if not self.updating_sliders and self.selected_track is not None:
                self._update_bbox_dimensions()

        @self.dim_1_slider.on_update
        def _(_):
            if not self.updating_sliders and self.selected_track is not None:
                self._update_bbox_dimensions()

        @self.dim_2_slider.on_update
        def _(_):
            if not self.updating_sliders and self.selected_track is not None:
                self._update_bbox_dimensions()

        # Elephant dimension mapping
        # User tells us which bbox dimension corresponds to which elephant body part
        self.server.add_gui_text("Elephant Axis Mapping", initial_value="Which bbox axis is the elephant's height?", disabled=True)

        self.height_is = self.server.add_gui_dropdown(
            "Height is",
            options=["dim[0]", "dim[1]", "dim[2]"],
            initial_value="dim[2]"
        )

        # Snap to proportions button
        self.snap_btn = self.server.add_gui_button("Snap to Elephant Proportions")

        @self.snap_btn.on_click
        def _(_):
            self._snap_to_proportions()

        # Copy from previous frame button
        copy_prev_btn = self.server.add_gui_button("Copy BBox from Previous Frame")

        @copy_prev_btn.on_click
        def _(_):
            self._copy_from_previous_frame()

        # Save button
        save_btn = self.server.add_gui_button("Save Corrections")

        @save_btn.on_click
        def _(_):
            self._save_corrections()

    def _render_frame(self):
        """Render current frame - full re-render"""
        # Clear all scene objects
        self._clear_scene()

        # Load and render point cloud
        points, colors = self._load_point_cloud(self.current_frame)
        if points is not None:
            point_size = self.point_size_slider.value if hasattr(self, 'point_size_slider') else 0.005
            self.pc_handle = self.server.add_point_cloud(
                name=f"/pc",
                points=points,
                colors=colors,
                point_size=point_size
            )

        # Get bboxes for current frame
        frame_bboxes = self.auto_bboxes.get(self.current_frame, [])

        if not frame_bboxes:
            self.info_text.value = f"Frame {self.current_frame} | No bboxes"
            return

        # Render each bbox
        for bbox in frame_bboxes:
            self._render_bbox(bbox)

        # Update selection (gizmo + sliders)
        self._update_selection()

        # Update info
        self.info_text.value = f"Frame {self.current_frame} | {len(frame_bboxes)} bboxes"

    def _render_bbox(self, bbox):
        """Render a single bbox wireframe"""
        # Use corrected version if exists
        if self._has_correction(bbox.track_id, self.current_frame):
            bbox = self.corrections[bbox.track_id][self.current_frame]
            color = (0.0, 1.0, 0.0)  # Green for corrected
        else:
            color = (1.0, 0.0, 0.0)  # Red for auto

        # Highlight selected
        if self.selected_track is not None and bbox.track_id == self.selected_track:
            color = (0.0, 1.0, 1.0)  # Cyan for selected
            line_width = 5.0
        else:
            line_width = 2.0

        # Create handle list for this bbox
        bbox_id = f"bbox_{bbox.track_id}"
        self.bbox_handles[bbox_id] = []

        # Render wireframe
        corners = bbox.get_corners()
        for i, (start_idx, end_idx) in enumerate(bbox.get_edges()):
            line_points = np.array([corners[start_idx], corners[end_idx]])

            handle = self.server.add_spline_catmull_rom(
                name=f"/{bbox_id}_edge_{i}",
                positions=line_points,
                color=color,
                line_width=line_width,
                segments=2
            )
            self.bbox_handles[bbox_id].append(handle)

        # Add center point
        handle = self.server.add_point_cloud(
            name=f"/{bbox_id}_center",
            points=bbox.center.reshape(1, 3),
            colors=np.array(color).reshape(1, 3),
            point_size=0.02
        )
        self.bbox_handles[bbox_id].append(handle)

    def _update_selection(self):
        """Update gizmo and sliders for selected bbox"""
        # Remove old gizmo
        if self.gizmo_handle is not None:
            self.gizmo_handle.remove()
            self.gizmo_handle = None

        if self.selected_track is None:
            return

        # Get selected bbox
        frame_bboxes = self.auto_bboxes.get(self.current_frame, [])
        selected_bbox = next((b for b in frame_bboxes if b.track_id == self.selected_track), None)

        if selected_bbox is None:
            self.info_text.value = f"Track {self.selected_track} not in frame {self.current_frame}"
            return

        # Use corrected version if exists
        if self._has_correction(self.selected_track, self.current_frame):
            selected_bbox = self.corrections[self.selected_track][self.current_frame]

        # Update sliders WITHOUT triggering callbacks
        self.updating_sliders = True
        self.dim_0_slider.value = float(selected_bbox.dimensions[0])
        self.dim_1_slider.value = float(selected_bbox.dimensions[1])
        self.dim_2_slider.value = float(selected_bbox.dimensions[2])
        self.updating_sliders = False

        # Create transform gizmo
        self.gizmo_handle = self.server.add_transform_controls(
            name=f"/gizmo",
            position=tuple(selected_bbox.center),
            wxyz=tuple(R.from_matrix(selected_bbox.rotation_matrix).as_quat()[[3, 0, 1, 2]])
        )

        @self.gizmo_handle.on_update
        def _(transform):
            self._update_bbox_from_gizmo(transform)

        # Update info
        self.info_text.value = f"Track {self.selected_track} | {selected_bbox.class_name} | Frame {self.current_frame}"

    def _clear_scene(self):
        """Clear all scene objects"""
        # Clear point cloud
        if self.pc_handle is not None:
            try:
                self.pc_handle.remove()
            except:
                pass
            self.pc_handle = None

        # Clear all bbox handles
        for bbox_id, handles in self.bbox_handles.items():
            for handle in handles:
                try:
                    handle.remove()
                except:
                    pass
        self.bbox_handles.clear()

        # Clear gizmo
        if self.gizmo_handle is not None:
            try:
                self.gizmo_handle.remove()
            except:
                pass
            self.gizmo_handle = None

    def _has_correction(self, track_id, frame_idx):
        """Check if correction exists"""
        return track_id in self.corrections and frame_idx in self.corrections[track_id]

    def _get_or_create_correction(self, track_id, frame_idx):
        """Get existing correction or create from original bbox"""
        if not self._has_correction(track_id, frame_idx):
            # Get original bbox
            frame_bboxes = self.auto_bboxes.get(frame_idx, [])
            original_bbox = next((b for b in frame_bboxes if b.track_id == track_id), None)
            if original_bbox is None:
                return None

            # Create correction entry
            if track_id not in self.corrections:
                self.corrections[track_id] = {}

            # Deep copy
            self.corrections[track_id][frame_idx] = BBox3D(
                center=original_bbox.center.copy(),
                dimensions=original_bbox.dimensions.copy(),
                rotation_matrix=original_bbox.rotation_matrix.copy(),
                class_name=original_bbox.class_name,
                track_id=original_bbox.track_id,
                frame_idx=original_bbox.frame_idx
            )

        return self.corrections[track_id][frame_idx]

    def _update_bbox_from_gizmo(self, transform):
        """Update bbox position/rotation from gizmo"""
        if self.selected_track is None:
            return

        # Get or create correction
        bbox = self._get_or_create_correction(self.selected_track, self.current_frame)
        if bbox is None:
            return

        # Update center
        bbox.center = np.array(transform.position)

        # Update rotation
        quat_wxyz = np.array(transform.wxyz)
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
        bbox.rotation_matrix = R.from_quat(quat_xyzw).as_matrix()

        # Re-render just the bboxes
        self._rerender_bboxes()

    def _update_bbox_dimensions(self):
        """Update bbox dimensions from sliders"""
        if self.selected_track is None:
            return

        # Get or create correction
        bbox = self._get_or_create_correction(self.selected_track, self.current_frame)
        if bbox is None:
            return

        # Update dimensions
        bbox.dimensions[0] = self.dim_0_slider.value
        bbox.dimensions[1] = self.dim_1_slider.value
        bbox.dimensions[2] = self.dim_2_slider.value

        # Re-render just the bboxes
        self._rerender_bboxes()

    def _snap_to_proportions(self):
        """Snap bbox to elephant proportions

        User tells us which dimension is height, we calculate the other two.
        Elephant proportions: Length/Height=1.72, Width/Height=0.78
        """
        if self.selected_track is None:
            return

        # Get or create correction
        bbox = self._get_or_create_correction(self.selected_track, self.current_frame)
        if bbox is None:
            return

        # Which dimension index is height? (0, 1, or 2)
        height_idx = int(self.height_is.value.split('[')[1].split(']')[0])

        # Get the other two indices
        all_indices = {0, 1, 2}
        other_indices = list(all_indices - {height_idx})

        # Get current height value
        height = bbox.dimensions[height_idx]

        # Calculate target length and width from elephant proportions
        target_length = height * 1.72
        target_width = height * 0.78

        # Assign to the other two dimensions
        # The larger dimension becomes length, smaller becomes width
        idx1, idx2 = other_indices[0], other_indices[1]
        dim1, dim2 = bbox.dimensions[idx1], bbox.dimensions[idx2]

        if dim1 > dim2:
            # idx1 is length, idx2 is width
            bbox.dimensions[idx1] = target_length
            bbox.dimensions[idx2] = target_width
        else:
            # idx2 is length, idx1 is width
            bbox.dimensions[idx2] = target_length
            bbox.dimensions[idx1] = target_width

        # Update sliders
        self.updating_sliders = True
        self.dim_0_slider.value = float(bbox.dimensions[0])
        self.dim_1_slider.value = float(bbox.dimensions[1])
        self.dim_2_slider.value = float(bbox.dimensions[2])
        self.updating_sliders = False

        # Re-render bboxes
        self._rerender_bboxes()

        print(f"✓ Snapped to elephant proportions:")
        print(f"  dim[0] = {bbox.dimensions[0]:.2f}m")
        print(f"  dim[1] = {bbox.dimensions[1]:.2f}m")
        print(f"  dim[2] = {bbox.dimensions[2]:.2f}m")
        print(f"  (Height was dim[{height_idx}] = {height:.2f}m)")

    def _copy_from_previous_frame(self):
        """Copy bbox from previous frame to current frame"""
        if self.selected_track is None:
            print("No track selected!")
            return

        if self.current_frame_idx == 0:
            print("Already at first frame!")
            return

        # Get previous frame
        prev_frame = self.frame_indices[self.current_frame_idx - 1]

        # Check if previous frame has this track (either corrected or original)
        prev_bbox = None
        if self._has_correction(self.selected_track, prev_frame):
            prev_bbox = self.corrections[self.selected_track][prev_frame]
        else:
            # Try to find in original bboxes
            prev_frame_bboxes = self.auto_bboxes.get(prev_frame, [])
            prev_bbox = next((b for b in prev_frame_bboxes if b.track_id == self.selected_track), None)

        if prev_bbox is None:
            print(f"Track {self.selected_track} not found in previous frame {prev_frame}!")
            return

        # Create correction for current frame by copying from previous
        if self.selected_track not in self.corrections:
            self.corrections[self.selected_track] = {}

        self.corrections[self.selected_track][self.current_frame] = BBox3D(
            center=prev_bbox.center.copy(),
            dimensions=prev_bbox.dimensions.copy(),
            rotation_matrix=prev_bbox.rotation_matrix.copy(),
            class_name=prev_bbox.class_name,
            track_id=prev_bbox.track_id,
            frame_idx=self.current_frame
        )

        print(f"✓ Copied bbox for track {self.selected_track} from frame {prev_frame} to {self.current_frame}")

        # Update sliders and re-render
        self._update_selection()
        self._rerender_bboxes()

    def _rerender_bboxes(self):
        """Re-render only the bboxes (faster than full re-render)"""
        # Clear bbox handles
        for bbox_id, handles in self.bbox_handles.items():
            for handle in handles:
                try:
                    handle.remove()
                except:
                    pass
        self.bbox_handles.clear()

        # Re-render all bboxes
        frame_bboxes = self.auto_bboxes.get(self.current_frame, [])
        for bbox in frame_bboxes:
            self._render_bbox(bbox)

    def _save_corrections(self):
        """Save corrections to JSON and generate 2D projections"""
        # 1. Save to results/corrected_labels/ (user edits only)
        corrected_labels_dir = self.results_dir / "corrected_labels"
        corrected_labels_dir.mkdir(exist_ok=True)

        output_file = corrected_labels_dir / "corrections.json"
        data = {}
        for track_id, frames in self.corrections.items():
            data[str(track_id)] = {}
            for frame_idx, bbox in frames.items():
                data[str(track_id)][str(frame_idx)] = {
                    'center': bbox.center.tolist(),
                    'dimensions': bbox.dimensions.tolist(),
                    'rotation_matrix': bbox.rotation_matrix.tolist(),
                    'class_name': bbox.class_name,
                    'track_id': bbox.track_id,
                    'frame_idx': bbox.frame_idx
                }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved {len(self.corrections)} track corrections to {output_file}")

        # 2. Save corrected bboxes to results/bounding_boxes_corrected/
        corrected_dir = self.results_dir / "bounding_boxes_corrected"
        corrected_dir.mkdir(exist_ok=True)

        # Build full bbox list per frame (corrected + original)
        all_frames = {}
        for frame_idx in self.frame_indices:
            # Start with original bboxes
            frame_bboxes = []
            for bbox in self.auto_bboxes.get(frame_idx, []):
                # Use corrected version if it exists
                if self._has_correction(bbox.track_id, frame_idx):
                    frame_bboxes.append(self.corrections[bbox.track_id][frame_idx])
                else:
                    frame_bboxes.append(bbox)
            all_frames[frame_idx] = frame_bboxes

        # Save each frame as JSON
        for frame_idx, frame_bboxes in all_frames.items():
            frame_file = corrected_dir / f"{frame_idx}.json"
            frame_data = []
            for bbox in frame_bboxes:
                frame_data.append({
                    'center': bbox.center.tolist(),
                    'dimensions': bbox.dimensions.tolist(),
                    'rotation_matrix': bbox.rotation_matrix.tolist(),
                    'class_name': bbox.class_name,
                    'track_id': bbox.track_id,
                    'frame_idx': bbox.frame_idx
                })
            with open(frame_file, 'w') as f:
                json.dump(frame_data, f, indent=2)

        print(f"✓ Saved corrected bboxes to {corrected_dir}")

        # 3. Generate 2D projections if images and camera params available
        if self.images_dir and self.cam_dict:
            self._generate_2d_projections(all_frames)
        else:
            print("⚠️ Skipping 2D projections (missing images or camera parameters)")

    def _generate_2d_projections(self, all_frames):
        """Generate 2D bbox projections on images"""
        import cv2

        print("\n📸 Generating 2D projections...")
        vis_dir = self.results_dir / "annotated_2d"
        vis_dir.mkdir(exist_ok=True)

        # Map frame indices to array indices
        frame_to_idx = {frame: idx for idx, frame in enumerate(self.frame_indices)}

        for frame_idx, frame_bboxes in all_frames.items():
            if frame_idx not in frame_to_idx:
                continue

            idx = frame_to_idx[frame_idx]

            # Load image
            img_file = self.images_dir / f"{frame_idx}.jpg"
            if not img_file.exists():
                img_file = self.images_dir / f"{frame_idx}.png"
            if not img_file.exists():
                continue

            img = cv2.imread(str(img_file))
            if img is None:
                continue

            # Get camera params
            focal = self.cam_dict['focal'][idx]
            pp = self.cam_dict['pp'][idx]
            R = self.cam_dict['R'][idx]
            t = self.cam_dict['t'][idx]

            # Scale camera params to image size
            img_h, img_w = img.shape[:2]
            model_h, model_w = 288, 512
            scale_x = img_w / model_w
            scale_y = img_h / model_h

            focal_scaled = focal * scale_x
            pp_scaled = pp * np.array([scale_x, scale_y])

            K = np.array([
                [focal_scaled, 0, pp_scaled[0]],
                [0, focal_scaled, pp_scaled[1]],
                [0, 0, 1]
            ])

            # Project each bbox
            for bbox in frame_bboxes:
                # Check if this bbox was corrected
                if self._has_correction(bbox.track_id, frame_idx):
                    color = (0, 255, 0)  # Green for corrected
                else:
                    color = (255, 0, 0)  # Blue for original

                # Get 3D corners
                corners_3d = bbox.get_corners()

                # Transform to camera space
                corners_cam = (R @ corners_3d.T).T + t

                # Project to 2D
                corners_2d = (K @ corners_cam.T).T
                corners_2d = corners_2d[:, :2] / corners_2d[:, 2:]

                # Draw wireframe
                corners_2d = corners_2d.astype(int)
                edges = bbox.get_edges()
                for start_idx, end_idx in edges:
                    pt1 = tuple(corners_2d[start_idx])
                    pt2 = tuple(corners_2d[end_idx])
                    cv2.line(img, pt1, pt2, color, 2)

                # Draw track ID
                center_2d = corners_2d.mean(axis=0).astype(int)
                cv2.putText(img, f"T{bbox.track_id}", tuple(center_2d),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save
            output_file = vis_dir / f"{frame_idx}_corrected.png"
            cv2.imwrite(str(output_file), img)

        print(f"✓ Saved 2D projections to {vis_dir}")

    def run(self):
        """Main loop"""
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nSaving...")
            self._save_corrections()
            print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto_bboxes", required=True, help="Dir with auto bbox JSONs")
    parser.add_argument("--output", required=True, help="Output dir for corrections")
    parser.add_argument("--images", help="Dir with images for 2D visualization (optional)")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")

    args = parser.parse_args()

    editor = SimpleBBoxEditor(
        auto_bboxes_dir=args.auto_bboxes,
        output_dir=args.output,
        images_dir=args.images,
        port=args.port
    )

    editor.run()


if __name__ == "__main__":
    main()


# - (a bit more complicated) we can have an additional option of select which face is the front, right and top for the current frame here instead of semantic propagation. Just an option