#!/usr/bin/env python3
"""
Viewpoint Analyzer v2 for WildLIFT Pipeline

Consumes output from the annotator tool (annotator_tool_v5.py) and performs
viewpoint characterization and inter-animal occlusion analysis.

This module provides TWO DISTINCT analyses:

1. VIEWPOINT ANALYSIS (visibility/coverage):
   - Coverage vector: How often each face is VISIBLE (points toward camera)
   - Diversity index: Are viewpoints evenly distributed?
   - Quality scores: When visible, how good is the view?
   - This is about CAMERA ANGLE relative to each animal

2. INTER-ANIMAL OCCLUSION ANALYSIS (physical blocking):
   - Occlusion rate: How often does one animal BLOCK another?
   - Per-track exposure: Which animals get blocked most often?
   - This is about DEPTH ORDERING when multiple animals are present

IMPORTANT DISTINCTION:
    - Face pointing away from camera = NOT VISIBLE (viewpoint geometry)
    - Face blocked by another animal = OCCLUDED (physical blocking)
    These are fundamentally different concepts!

Usage:
    python viewpoint_analyzer_v2.py --annotator_output results/zebra/scene1/corrected/
    python viewpoint_analyzer_v2.py --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/ --compute_occlusion
"""

import os
import json
import numpy as np
import cv2
import glob
from pathlib import Path
import re
import colorsys
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from scipy.stats import entropy

# Optional dependencies for occlusion analysis
try:
    from scipy.spatial import ConvexHull
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. ConvexHull operations will be limited.")

try:
    from shapely.geometry import Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: shapely not available. Inter-animal occlusion will not be computed.")


class ViewpointAnalyzer:
    """
    Viewpoint and occlusion analysis for annotated wildlife tracklets.

    Consumes pre-annotated data from the annotator tool and computes
    viewpoint metrics and occlusion statistics without interactive labeling.
    """

    def __init__(self, annotator_output_dir, images_dir=None):
        """
        Initialize viewpoint analyzer with annotator tool output.

        Args:
            annotator_output_dir: Path to annotator tool output directory.
                Expected structure:
                    annotator_output_dir/
                    ├── bounding_boxes/*.json
                    ├── corrected_labels/semantic_faces/manual_labels.json
                    └── camera/*.npz (or symlink to results dir)

            images_dir: Optional path to original images for visualization.
                If None, visualizations requiring images will be skipped.

        Raises:
            FileNotFoundError: If required annotation files are missing.
        """
        self.annotator_output_dir = Path(annotator_output_dir)
        self.images_dir = Path(images_dir) if images_dir else None

        # Validate required paths exist
        self._validate_input_paths()

        # Load corrected bboxes with semantic face annotations
        self.all_bbox_data = self._load_annotated_bboxes()
        self.semantic_faces = self._load_semantic_faces()

        # Derive labeled tracks from available annotations
        self.labeled_tracks = list(self.semantic_faces.keys())

        # Establish temporal ordering
        self.frame_order = self._determine_frame_order()

        # Visualization configuration
        self.semantic_face_colors = {
            'front': '#FF4444',   # Red
            'back': '#44FF44',    # Green
            'left': '#4444FF',    # Blue
            'right': '#FFAA44',   # Orange
            'top': '#FF44FF',     # Magenta
            'bottom': '#44FFFF',  # Cyan
        }

        # Track colors for visualization (BGR format)
        self.track_colors = {
            0: (0, 255, 255),    # Cyan
            1: (255, 0, 255),    # Magenta
            2: (255, 255, 0),    # Yellow
            3: (0, 255, 0),      # Green
            4: (255, 128, 0),    # Orange
            5: (128, 0, 255),    # Purple
            6: (0, 128, 255),    # Sky Blue
            7: (255, 0, 128),    # Pink
            8: (128, 255, 0),    # Lime
            9: (0, 255, 128),    # Spring Green
        }

        print(f"Viewpoint Analyzer v2 initialized:")
        print(f"  Annotator output: {self.annotator_output_dir}")
        print(f"  Tracks with semantic labels: {self.labeled_tracks}")
        print(f"  Total frames: {len(self.frame_order)}")
        print(f"  Occlusion analysis available: {SCIPY_AVAILABLE and SHAPELY_AVAILABLE}")

    def _validate_input_paths(self):
        """Validate that required annotator output files exist."""
        bbox_dir = self.annotator_output_dir / "bounding_boxes"
        semantic_file = (self.annotator_output_dir /
                        "corrected_labels" / "semantic_faces" / "manual_labels.json")

        errors = []

        if not bbox_dir.exists():
            errors.append(f"Bounding boxes directory not found: {bbox_dir}")
        elif not list(bbox_dir.glob("*.json")):
            errors.append(f"No bounding box JSON files in: {bbox_dir}")

        if not semantic_file.exists():
            errors.append(
                f"Semantic face labels not found: {semantic_file}\n"
                "  Run annotator tool first and label semantic faces for at least one track."
            )

        if errors:
            raise FileNotFoundError("\n".join(errors))

    def _load_annotated_bboxes(self):
        """
        Load bounding boxes from annotator tool output.

        These are the CORRECTED bboxes (user-refined positions, dimensions,
        and rotations) rather than raw auto-generated bboxes.

        Returns:
            Dict[int, Dict[str, dict]]: Mapping of track_id -> frame_name -> bbox_data
        """
        bbox_dir = self.annotator_output_dir / "bounding_boxes"
        all_bbox_data = {}

        for bbox_file in sorted(bbox_dir.glob("*.json")):
            frame_name = bbox_file.stem

            with open(bbox_file, 'r') as f:
                frame_bboxes = json.load(f)

            for bbox in frame_bboxes:
                track_id = bbox.get('track_id')

                # Skip untracked detections
                if track_id is None or track_id == -1:
                    continue

                if track_id not in all_bbox_data:
                    all_bbox_data[track_id] = {}

                all_bbox_data[track_id][frame_name] = {
                    'center': np.array(bbox['center']),
                    'dimensions': np.array(bbox['dimensions']),
                    'rotation_matrix': np.array(bbox['rotation_matrix']),
                    'track_id': track_id,
                    'class_name': bbox.get('class_name', 'animal'),
                    'confidence': bbox.get('confidence', 1.0)
                }

        total_frames = sum(len(frames) for frames in all_bbox_data.values())
        print(f"  Loaded bboxes: {len(all_bbox_data)} tracks, {total_frames} frame-instances")

        return all_bbox_data

    def _load_semantic_faces(self):
        """
        Load semantic face labels from annotator tool output.

        The annotator tool saves labels in format:
            {track_id_str: {frame_name: {semantic_label: face_index}}}

        This method converts face indices back to face geometry data
        by reconstructing from the corresponding bbox.

        Returns:
            Dict[int, Dict[str, Dict[str, dict]]]:
                track_id -> frame_name -> {semantic_label -> face_data}
        """
        semantic_file = (self.annotator_output_dir /
                        "corrected_labels" / "semantic_faces" / "manual_labels.json")

        with open(semantic_file, 'r') as f:
            raw_labels = json.load(f)

        semantic_faces = {}

        for track_id_str, frames_data in raw_labels.items():
            track_id = int(track_id_str)

            if track_id not in self.all_bbox_data:
                print(f"  Warning: Labels exist for track {track_id} but no bbox data found")
                continue

            semantic_faces[track_id] = {}

            for frame_name, label_to_index in frames_data.items():
                # Handle both string and int frame names
                frame_key = str(frame_name)

                if frame_key not in self.all_bbox_data[track_id]:
                    print(f"  Warning: Labels for track {track_id} frame {frame_key} "
                          "but no bbox data")
                    continue

                # Get bbox and compute all 6 faces
                bbox_data = self.all_bbox_data[track_id][frame_key]
                all_faces = self.get_all_faces_from_bbox(bbox_data)

                # Map semantic labels to face geometry
                semantic_faces[track_id][frame_key] = {}

                for semantic_label, face_index in label_to_index.items():
                    face_key = f'f{face_index}'

                    if face_key in all_faces:
                        semantic_faces[track_id][frame_key][semantic_label] = all_faces[face_key]
                    else:
                        print(f"  Warning: Face index {face_index} invalid for "
                              f"track {track_id} frame {frame_key}")

                # Infer opposite faces (back from front, right from left, bottom from top)
                self._infer_opposite_faces(semantic_faces[track_id][frame_key], all_faces)

        total_labels = sum(
            len(frames) for frames in semantic_faces.values()
        )
        print(f"  Loaded semantic labels: {len(semantic_faces)} tracks, "
              f"{total_labels} frame-label sets")

        return semantic_faces

    def _determine_frame_order(self):
        """Determine the temporal order of frames."""
        all_frame_names = set()
        for track_data in self.all_bbox_data.values():
            all_frame_names.update(track_data.keys())

        def extract_numeric_part(frame_name):
            numbers = re.findall(r'\d+', str(frame_name))
            return int(numbers[0]) if numbers else float('inf')

        sorted_frames = sorted(list(all_frame_names), key=extract_numeric_part)
        return sorted_frames

    def _load_camera_params(self, frame_name):
        """
        Load camera parameters for a specific frame.

        Searches in order:
            1. annotator_output_dir/camera/{frame_name}.npz
            2. Parent results directory/camera/{frame_name}.npz

        Args:
            frame_name: Frame identifier (without extension)

        Returns:
            Dict with 'K' (intrinsics), 'R' (rotation), 't' (translation)
            or None if camera file not found
        """
        # Primary location: annotator output
        camera_file = self.annotator_output_dir / "camera" / f"{frame_name}.npz"

        # Fallback: parent results directory (annotator may not copy camera files)
        if not camera_file.exists():
            results_dir = self.annotator_output_dir.parent
            camera_file = results_dir / "camera" / f"{frame_name}.npz"

        # Second fallback: sibling directory structure
        if not camera_file.exists():
            # Try going up multiple levels
            for parent in [self.annotator_output_dir.parent,
                          self.annotator_output_dir.parent.parent]:
                camera_file = parent / "camera" / f"{frame_name}.npz"
                if camera_file.exists():
                    break

        if not camera_file.exists():
            return None

        try:
            camera_data = np.load(camera_file)
            return {
                'K': camera_data['intrinsics'],
                'R': camera_data['pose'][:3, :3],
                't': camera_data['pose'][:3, 3]
            }
        except Exception as e:
            print(f"  Warning: Failed to load camera params for {frame_name}: {e}")
            return None

    # ========================================================================
    # GEOMETRY METHODS
    # ========================================================================

    def get_bbox_corners(self, center, dimensions, rotation_matrix):
        """Get 8 corner points of a 3D bounding box."""
        l, w, h = dimensions

        corners_local = np.array([
            [-l/2, -w/2, -h/2],  # 0: bottom-back-left
            [+l/2, -w/2, -h/2],  # 1: bottom-back-right
            [+l/2, +w/2, -h/2],  # 2: bottom-front-right
            [-l/2, +w/2, -h/2],  # 3: bottom-front-left
            [-l/2, -w/2, +h/2],  # 4: top-back-left
            [+l/2, -w/2, +h/2],  # 5: top-back-right
            [+l/2, +w/2, +h/2],  # 6: top-front-right
            [-l/2, +w/2, +h/2],  # 7: top-front-left
        ])

        corners_world = (rotation_matrix @ corners_local.T).T + center
        return corners_world

    def compute_face_from_corners(self, corners, indices, box_center):
        """Compute face properties from corner indices."""
        face_corners = corners[indices]
        face_center = np.mean(face_corners, axis=0)

        edge1 = face_corners[1] - face_corners[0]
        edge2 = face_corners[3] - face_corners[0]
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)

        outward_vec = face_center - box_center
        if np.dot(normal, outward_vec) < 0:
            normal = -normal

        area = 0.5 * (np.linalg.norm(np.cross(edge1, edge2)) +
                     np.linalg.norm(np.cross(face_corners[2] - face_corners[0],
                                           face_corners[3] - face_corners[0])))

        return {
            'center': face_center,
            'normal': normal,
            'corners': face_corners,
            'area': area
        }

    def get_all_faces_from_bbox(self, bbox_data):
        """Get all 6 faces from bbox data.

        IMPORTANT: Face indices must match annotator_tool_v5.py definition:
            Face 0: Front (along +X)
            Face 1: Back (along -X)
            Face 2: Left (along -Y)
            Face 3: Right (along +Y)
            Face 4: Top (along +Z)
            Face 5: Bottom (along -Z)
        """
        corners = self.get_bbox_corners(
            bbox_data['center'],
            bbox_data['dimensions'],
            bbox_data['rotation_matrix']
        )
        box_center = np.mean(corners, axis=0)

        # Face indices matching annotator_tool_v5.py
        face_indices = {
            'f0': [0, 1, 5, 4],  # Front (along +X)
            'f1': [2, 3, 7, 6],  # Back (along -X)
            'f2': [0, 3, 7, 4],  # Left (along -Y)
            'f3': [1, 2, 6, 5],  # Right (along +Y)
            'f4': [4, 5, 6, 7],  # Top (along +Z)
            'f5': [0, 1, 2, 3],  # Bottom (along -Z)
        }

        faces = {}
        for face_id, indices in face_indices.items():
            faces[face_id] = self.compute_face_from_corners(corners, indices, box_center)

        return faces

    def _infer_opposite_faces(self, semantic_assignments, all_faces):
        """Infer opposite faces based on normal similarity."""
        opposites = {'front': 'back', 'left': 'right', 'top': 'bottom'}

        # Track which faces are already assigned
        assigned_faces = set()
        for semantic_label, face_data in semantic_assignments.items():
            for face_id, test_face in all_faces.items():
                if np.allclose(face_data['center'], test_face['center']):
                    assigned_faces.add(face_id)
                    break

        unassigned_faces = {fid: fdata for fid, fdata in all_faces.items()
                           if fid not in assigned_faces}

        for semantic_label, face_data in list(semantic_assignments.items()):
            if semantic_label in opposites:
                opposite_label = opposites[semantic_label]

                # Skip if opposite already assigned
                if opposite_label in semantic_assignments:
                    continue

                best_match = None
                best_score = 999

                for face_id, test_face in unassigned_faces.items():
                    # Opposite faces should have anti-parallel normals
                    dot = np.dot(face_data['normal'], test_face['normal'])
                    if dot < best_score:
                        best_score = dot
                        best_match = test_face
                        best_match_id = face_id

                if best_match is not None and best_score < -0.8:
                    semantic_assignments[opposite_label] = best_match
                    if best_match_id in unassigned_faces:
                        del unassigned_faces[best_match_id]

    def get_track_color(self, track_id):
        """Get color for a track."""
        if track_id in self.track_colors:
            return self.track_colors[track_id]
        else:
            hue = (track_id * 0.618033988749895) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
            return (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))

    # ========================================================================
    # VISIBILITY AND QUALITY METHODS
    # ========================================================================

    def _project_face_to_2d(self, face_data, camera_params, img_shape):
        """Project a 3D face to 2D image."""
        try:
            face_corners_3d = face_data['corners']

            K = camera_params['K']
            R = camera_params['R']
            t = camera_params['t']

            camera_pose = np.eye(4)
            camera_pose[:3, :3] = R
            camera_pose[:3, 3] = t

            face_corners_h = np.concatenate([face_corners_3d, np.ones((4, 1))], axis=1)
            corners_cam = (np.linalg.inv(camera_pose) @ face_corners_h.T).T[:, :3]

            if np.any(corners_cam[:, 2] <= 0):
                return None

            corners_2d_hom = (K @ corners_cam.T).T
            corners_2d = corners_2d_hom[:, :2] / corners_2d_hom[:, 2:3]

            img_h, img_w = img_shape[:2]
            if (np.any(corners_2d[:, 0] < -img_w) or np.any(corners_2d[:, 0] > 2*img_w) or
                np.any(corners_2d[:, 1] < -img_h) or np.any(corners_2d[:, 1] > 2*img_h)):
                return None

            return corners_2d.astype(int)
        except:
            return None

    def _calculate_face_visibility(self, face_data, camera_params):
        """Calculate if a semantic face is visible to the camera."""
        try:
            # Direct translation for camera-to-world pose convention
            t = camera_params['t']
            camera_pos = t  # No inversion needed

            # Vector from face to camera
            face_to_camera = camera_pos - face_data['center']
            face_to_camera = face_to_camera / np.linalg.norm(face_to_camera)

            # Dot product with face normal (positive = facing camera)
            visibility_score = np.dot(face_data['normal'], face_to_camera)

            return visibility_score > 0, visibility_score
        except:
            return False, 0.0

    def _calculate_face_quality(self, face_data, camera_params, img_shape):
        """Calculate comprehensive face quality score for frame ranking."""
        try:
            # Visibility component
            is_visible, visibility_score = self._calculate_face_visibility(face_data, camera_params)
            if not is_visible:
                return 0.0

            # Projection quality
            corners_2d = self._project_face_to_2d(face_data, camera_params, img_shape)
            if corners_2d is None:
                return 0.0

            # Face area in image (larger = better for feature extraction)
            face_area_2d = cv2.contourArea(corners_2d)
            max_possible_area = img_shape[0] * img_shape[1]
            area_score = min(face_area_2d / max_possible_area, 1.0)

            # Distance from center (closer to center = better)
            img_center = np.array([img_shape[1]/2, img_shape[0]/2])
            face_center_2d = np.mean(corners_2d, axis=0)
            max_distance = np.sqrt(img_shape[0]**2 + img_shape[1]**2) / 2
            distance_score = 1.0 - np.linalg.norm(face_center_2d - img_center) / max_distance

            # Aspect ratio score (closer to square = better)
            x_span = np.max(corners_2d[:, 0]) - np.min(corners_2d[:, 0])
            y_span = np.max(corners_2d[:, 1]) - np.min(corners_2d[:, 1])
            if min(x_span, y_span) > 0:
                aspect_ratio = min(x_span, y_span) / max(x_span, y_span)
            else:
                aspect_ratio = 0.0

            # Combined quality score
            quality_score = (
                0.4 * abs(visibility_score) +  # Face orientation toward camera
                0.3 * area_score +              # Size in image
                0.2 * distance_score +          # Central positioning
                0.1 * aspect_ratio              # Shape preservation
            )

            return quality_score
        except:
            return 0.0

    def _highlight_projected_face(self, img, corners_2d, color, thickness=3, alpha=0.3):
        """Draw face on image."""
        if corners_2d is None or len(corners_2d) != 4:
            return

        overlay = img.copy()
        pts = corners_2d.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

        for corner in corners_2d:
            cv2.circle(img, tuple(corner), 6, color, -1)

    # ========================================================================
    # SEMANTIC CONSISTENCY VALIDATION
    # ========================================================================

    def validate_semantic_consistency(self):
        """
        Check that semantic face labels are geometrically consistent across frames.

        Validates:
            1. Face normals don't flip between consecutive frames
            2. Inferred opposite faces are actually opposite
            3. Front/left/top form a valid right-handed coordinate system

        Returns:
            List[Dict]: List of detected issues, each containing:
                - track: Track ID
                - frames: Tuple of (prev_frame, curr_frame)
                - face: Semantic label with issue
                - issue_type: 'normal_flip', 'not_opposite', 'chirality'
                - severity: float in [0, 1]
        """
        issues = []

        for track_id in self.labeled_tracks:
            if track_id not in self.semantic_faces:
                continue

            frames = sorted(
                self.semantic_faces[track_id].keys(),
                key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0
            )

            # Check consecutive frame consistency
            for i in range(1, len(frames)):
                prev_frame = frames[i-1]
                curr_frame = frames[i]

                prev_faces = self.semantic_faces[track_id][prev_frame]
                curr_faces = self.semantic_faces[track_id][curr_frame]

                # Check each semantic label
                for label in ['front', 'left', 'top']:
                    if label not in prev_faces or label not in curr_faces:
                        continue

                    prev_normal = prev_faces[label]['normal']
                    curr_normal = curr_faces[label]['normal']

                    dot_product = np.dot(prev_normal, curr_normal)

                    # Normal should be roughly consistent (allowing for rotation)
                    if dot_product < 0.5:  # More than ~60 degree change
                        issues.append({
                            'track': track_id,
                            'frames': (prev_frame, curr_frame),
                            'face': label,
                            'issue_type': 'normal_flip',
                            'severity': 1.0 - max(0, dot_product),
                            'details': f'Normal dot product: {dot_product:.3f}'
                        })

                # Check opposite face consistency
                opposites = [('front', 'back'), ('left', 'right'), ('top', 'bottom')]
                for primary, opposite in opposites:
                    if primary in curr_faces and opposite in curr_faces:
                        primary_normal = curr_faces[primary]['normal']
                        opposite_normal = curr_faces[opposite]['normal']

                        dot = np.dot(primary_normal, opposite_normal)

                        # Opposite faces should have anti-parallel normals
                        if dot > -0.8:  # Not sufficiently opposite
                            issues.append({
                                'track': track_id,
                                'frames': (curr_frame, curr_frame),
                                'face': f'{primary}/{opposite}',
                                'issue_type': 'not_opposite',
                                'severity': (dot + 1) / 2,  # Map [-1,1] to [0,1]
                                'details': f'Opposite normals dot: {dot:.3f}'
                            })

        return issues

    # ========================================================================
    # OCCLUSION ANALYSIS METHODS
    # ========================================================================

    def _get_bboxes_for_frame(self, frame_name):
        """
        Get all bounding boxes present in a specific frame.

        Args:
            frame_name: Frame identifier

        Returns:
            List[Dict]: List of bbox_data dicts with track_id included
        """
        frame_bboxes = []

        for track_id, track_data in self.all_bbox_data.items():
            if frame_name in track_data:
                bbox_data = track_data[frame_name].copy()
                bbox_data['track_id'] = track_id
                frame_bboxes.append(bbox_data)

        return frame_bboxes

    def _project_bbox_to_2d(self, bbox_data, camera_params):
        """
        Project 3D OBB corners to 2D image coordinates.

        Args:
            bbox_data: Dict with center, dimensions, rotation_matrix
            camera_params: Dict with K, R, t

        Returns:
            np.ndarray (8, 2): 2D corner coordinates, or None if behind camera
        """
        corners_3d = self.get_bbox_corners(
            bbox_data['center'],
            bbox_data['dimensions'],
            bbox_data['rotation_matrix']
        )

        K = camera_params['K']
        R = camera_params['R']
        t = camera_params['t']

        # Build camera pose matrix
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R
        camera_pose[:3, 3] = t

        # Transform to camera coordinates
        corners_h = np.concatenate([corners_3d, np.ones((8, 1))], axis=1)
        corners_cam = (np.linalg.inv(camera_pose) @ corners_h.T).T[:, :3]

        # Check for points behind camera
        if np.any(corners_cam[:, 2] <= 0):
            return None

        # Project to image plane
        corners_2d_h = (K @ corners_cam.T).T
        corners_2d = corners_2d_h[:, :2] / corners_2d_h[:, 2:3]

        return corners_2d

    def _compute_2d_overlap(self, corners_a, corners_b):
        """
        Compute fraction of hull_b's area overlapped by hull_a.

        Args:
            corners_a: np.ndarray (N, 2) - 2D points of first bbox
            corners_b: np.ndarray (M, 2) - 2D points of second bbox

        Returns:
            float: Overlap fraction in [0, 1] = Area(intersection) / Area(hull_b)
        """
        if not SCIPY_AVAILABLE or not SHAPELY_AVAILABLE:
            return 0.0

        try:
            # Compute convex hulls
            hull_a = ConvexHull(corners_a)
            hull_b = ConvexHull(corners_b)

            poly_a = Polygon(corners_a[hull_a.vertices])
            poly_b = Polygon(corners_b[hull_b.vertices])

            if not poly_a.is_valid or not poly_b.is_valid:
                return 0.0

            intersection = poly_a.intersection(poly_b)
            area_b = poly_b.area

            if area_b <= 0:
                return 0.0

            return intersection.area / area_b

        except Exception as e:
            # Fallback if computation fails
            return 0.0

    def compute_inter_animal_occlusion(self, frame_name):
        """
        Compute pairwise occlusion matrix for all animals in a frame.

        For each pair of animals (A, B), computes what fraction of B's
        2D projection is occluded by A, considering depth ordering.

        Args:
            frame_name: Frame identifier

        Returns:
            Dict containing:
                - 'matrix': np.ndarray (N, N) occlusion values
                - 'track_ids': List of track IDs (row/column order)
                - 'num_occluded_pairs': Count of pairs with occlusion > threshold
                - 'max_occlusion': Maximum occlusion value in frame
        """
        camera_params = self._load_camera_params(frame_name)
        if camera_params is None:
            return None

        frame_bboxes = self._get_bboxes_for_frame(frame_name)
        n = len(frame_bboxes)

        if n < 2:
            return {
                'matrix': np.zeros((n, n)),
                'track_ids': [b['track_id'] for b in frame_bboxes],
                'num_occluded_pairs': 0,
                'max_occlusion': 0.0
            }

        # Extract camera position
        camera_pos = camera_params['t']

        # Initialize occlusion matrix
        occlusion_matrix = np.zeros((n, n))
        track_ids = [b['track_id'] for b in frame_bboxes]

        # Project all bboxes to 2D
        projections = []
        distances = []

        for bbox in frame_bboxes:
            corners_2d = self._project_bbox_to_2d(bbox, camera_params)
            projections.append(corners_2d)

            dist = np.linalg.norm(bbox['center'] - camera_pos)
            distances.append(dist)

        # Compute pairwise occlusion
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                # Skip if projection failed for either
                if projections[i] is None or projections[j] is None:
                    continue

                # i can only occlude j if i is closer to camera
                if distances[i] >= distances[j]:
                    continue

                # Compute 2D overlap (fraction of j occluded by i)
                overlap = self._compute_2d_overlap(projections[i], projections[j])
                occlusion_matrix[i, j] = overlap

        # Summary statistics
        THRESHOLD = 0.1
        num_occluded = np.sum(occlusion_matrix > THRESHOLD)
        max_occ = np.max(occlusion_matrix) if n > 0 else 0.0

        return {
            'matrix': occlusion_matrix,
            'track_ids': track_ids,
            'num_occluded_pairs': int(num_occluded),
            'max_occlusion': float(max_occ)
        }

    def _empty_occlusion_stats(self):
        """Return empty occlusion stats structure."""
        return {
            'occlusion_rate': 0.0,
            'mean_severity': 0.0,
            'max_severity': 0.0,
            'per_track_exposure': {},
            'temporal_profile': [],
            'total_frames_analyzed': 0,
            'frames_with_multiple_animals': 0,
            'frames_with_occlusion': 0
        }

    def compute_occlusion_statistics(self):
        """
        Compute inter-animal occlusion statistics across all frames.

        NOTE: This computes ONLY inter-animal occlusion (one animal blocking another).
        Face visibility based on normal direction is computed separately in
        viewpoint analysis (coverage_vector, visibility scores).

        Returns:
            Dict containing:
                - 'occlusion_rate': Fraction of multi-animal frames with blocking
                - 'mean_severity': Mean overlap when blocking occurs
                - 'max_severity': Maximum overlap observed
                - 'per_track_exposure': Dict[track_id -> mean occlusion received]
                - 'temporal_profile': List of per-frame summaries
        """
        print("\n" + "="*70)
        print("COMPUTING INTER-ANIMAL OCCLUSION STATISTICS")
        print("="*70)

        if not SCIPY_AVAILABLE or not SHAPELY_AVAILABLE:
            print("  Warning: scipy/shapely not available. Cannot compute occlusion.")
            return self._empty_occlusion_stats()

        OCCLUSION_THRESHOLD = 0.1

        stats = {
            'occlusion_rate': 0.0,
            'mean_severity': 0.0,
            'max_severity': 0.0,
            'per_track_exposure': defaultdict(list),
            'temporal_profile': [],
            'total_frames_analyzed': 0,
            'frames_with_multiple_animals': 0,
            'frames_with_occlusion': 0
        }

        all_occlusion_values = []

        for frame_idx, frame_name in enumerate(self.frame_order):
            if frame_idx % 50 == 0:
                print(f"  Processing frame {frame_idx}/{len(self.frame_order)}...")

            # Compute inter-animal occlusion matrix for this frame
            occ_result = self.compute_inter_animal_occlusion(frame_name)

            if occ_result is None:
                continue

            stats['total_frames_analyzed'] += 1
            n_animals = len(occ_result['track_ids'])

            # Skip frames with only one animal (no occlusion possible)
            if n_animals < 2:
                stats['temporal_profile'].append({
                    'frame': frame_name,
                    'num_animals': n_animals,
                    'num_occluded_pairs': 0,
                    'max_occlusion': 0.0,
                    'occluder_occludee_pairs': []
                })
                continue

            stats['frames_with_multiple_animals'] += 1

            occ_matrix = occ_result['matrix']
            track_ids = occ_result['track_ids']

            # Check if any occlusion occurred
            if occ_result['num_occluded_pairs'] > 0:
                stats['frames_with_occlusion'] += 1

            # Collect per-track exposure (how much each track gets blocked)
            # This is the sum of column j = total occlusion received by track j
            for j, track_id in enumerate(track_ids):
                occlusion_received = np.sum(occ_matrix[:, j])
                if occlusion_received > 0:
                    stats['per_track_exposure'][track_id].append(occlusion_received)
                    if occlusion_received > OCCLUSION_THRESHOLD:
                        all_occlusion_values.append(occlusion_received)

            # Record occluder-occludee pairs for this frame
            pairs = []
            for i in range(n_animals):
                for j in range(n_animals):
                    if occ_matrix[i, j] > OCCLUSION_THRESHOLD:
                        pairs.append({
                            'occluder': track_ids[i],
                            'occludee': track_ids[j],
                            'severity': float(occ_matrix[i, j])
                        })

            stats['temporal_profile'].append({
                'frame': frame_name,
                'num_animals': n_animals,
                'num_occluded_pairs': occ_result['num_occluded_pairs'],
                'max_occlusion': occ_result['max_occlusion'],
                'occluder_occludee_pairs': pairs
            })

        # Aggregate statistics
        if stats['frames_with_multiple_animals'] > 0:
            stats['occlusion_rate'] = (
                stats['frames_with_occlusion'] / stats['frames_with_multiple_animals']
            )

        if all_occlusion_values:
            stats['mean_severity'] = float(np.mean(all_occlusion_values))
            stats['max_severity'] = float(np.max(all_occlusion_values))

        # Convert per-track exposure to mean values
        stats['per_track_exposure'] = {
            track_id: float(np.mean(values)) if values else 0.0
            for track_id, values in stats['per_track_exposure'].items()
        }

        # Print summary
        print(f"\n  Frames analyzed: {stats['total_frames_analyzed']}")
        print(f"  Frames with 2+ animals: {stats['frames_with_multiple_animals']}")
        print(f"  Frames with occlusion: {stats['frames_with_occlusion']}")
        print(f"  Occlusion rate: {stats['occlusion_rate']:.1%}")
        print(f"  Mean severity: {stats['mean_severity']:.3f}")
        print(f"  Max severity: {stats['max_severity']:.3f}")

        if stats['per_track_exposure']:
            print(f"\n  Per-track occlusion exposure:")
            for track_id, exposure in sorted(stats['per_track_exposure'].items()):
                print(f"    Track {track_id}: {exposure:.3f}")

        return stats

    def create_occlusion_visualizations(self, occlusion_stats=None):
        """
        Generate inter-animal occlusion visualizations.

        NOTE: Does NOT include "self-occlusion" or per-face visibility.
        Those are viewpoint metrics, shown in the viewpoint dashboard.

        Creates:
            1. Temporal occlusion timeline
            2. Per-track occlusion exposure bar chart
            3. Occlusion severity distribution histogram
            4. Summary statistics

        Args:
            occlusion_stats: Pre-computed stats, or None to compute

        Returns:
            Path to output directory
        """
        print("\n" + "="*70)
        print("GENERATING INTER-ANIMAL OCCLUSION VISUALIZATIONS")
        print("="*70)

        viz_dir = self.annotator_output_dir / "viewpoint_analysis"
        viz_dir.mkdir(exist_ok=True)

        if occlusion_stats is None:
            occlusion_stats = self.compute_occlusion_statistics()

        fig = plt.figure(figsize=(14, 10))

        # ===== Panel 1: Temporal Occlusion Timeline =====
        ax1 = plt.subplot(2, 2, 1)

        temporal = occlusion_stats['temporal_profile']
        if temporal:
            # Only include frames with 2+ animals
            multi_animal_frames = [t for t in temporal if t['num_animals'] >= 2]

            if multi_animal_frames:
                frame_indices = range(len(multi_animal_frames))
                max_occs = [t['max_occlusion'] for t in multi_animal_frames]

                ax1.fill_between(frame_indices, max_occs, alpha=0.3, color='red')
                ax1.plot(frame_indices, max_occs, color='red', linewidth=1)

                # Mark severe occlusion events
                severe_threshold = 0.3
                severe_frames = [i for i, occ in enumerate(max_occs) if occ > severe_threshold]
                if severe_frames:
                    ax1.scatter(severe_frames, [max_occs[i] for i in severe_frames],
                               color='darkred', s=30, zorder=5, label=f'Severe (>{severe_threshold:.0%})')
                    ax1.legend(loc='upper right')

                ax1.set_xlabel('Multi-Animal Frame Index', fontsize=11)
                ax1.set_ylabel('Max Occlusion Severity', fontsize=11)
                ax1.set_title('Inter-Animal Occlusion Over Time', fontsize=13, fontweight='bold')
                ax1.set_ylim(0, 1)
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No multi-animal frames', ha='center', va='center')
                ax1.set_title('Inter-Animal Occlusion Over Time', fontsize=13, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax1.set_title('Inter-Animal Occlusion Over Time', fontsize=13, fontweight='bold')

        # ===== Panel 2: Per-Track Occlusion Exposure =====
        ax2 = plt.subplot(2, 2, 2)

        per_track = occlusion_stats['per_track_exposure']
        if per_track:
            track_ids = sorted(per_track.keys())
            exposure_values = [per_track[tid] for tid in track_ids]

            max_val = max(exposure_values) if exposure_values and max(exposure_values) > 0 else 1
            colors = plt.cm.Reds([v / max_val for v in exposure_values])

            bars = ax2.bar(range(len(track_ids)), exposure_values, color=colors)
            ax2.set_xticks(range(len(track_ids)))
            ax2.set_xticklabels([f'T{tid}' for tid in track_ids])
            ax2.set_xlabel('Track ID', fontsize=11)
            ax2.set_ylabel('Mean Occlusion Exposure', fontsize=11)
            ax2.set_title('How Often Each Animal Gets Blocked', fontsize=13, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)

            for bar, val in zip(bars, exposure_values):
                if val > 0.01:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No inter-animal occlusion detected',
                    ha='center', va='center', fontsize=12)
            ax2.set_title('How Often Each Animal Gets Blocked', fontsize=13, fontweight='bold')

        # ===== Panel 3: Occlusion Event Histogram =====
        ax3 = plt.subplot(2, 2, 3)

        if temporal:
            occlusion_values = []
            for t in temporal:
                for pair in t.get('occluder_occludee_pairs', []):
                    occlusion_values.append(pair['severity'])

            if occlusion_values:
                ax3.hist(occlusion_values, bins=20, range=(0, 1),
                        color='coral', edgecolor='darkred', alpha=0.7)
                ax3.axvline(x=np.mean(occlusion_values), color='red', linestyle='--',
                           linewidth=2, label=f'Mean: {np.mean(occlusion_values):.2f}')
                ax3.set_xlabel('Occlusion Severity', fontsize=11)
                ax3.set_ylabel('Number of Occlusion Events', fontsize=11)
                ax3.set_title('Distribution of Occlusion Severity', fontsize=13, fontweight='bold')
                ax3.legend()
                ax3.grid(axis='y', alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No occlusion events', ha='center', va='center')
                ax3.set_title('Distribution of Occlusion Severity', fontsize=13, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax3.set_title('Distribution of Occlusion Severity', fontsize=13, fontweight='bold')

        # ===== Panel 4: Summary Statistics =====
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')

        frames_with_occlusion = occlusion_stats.get('frames_with_occlusion', 0)

        summary_lines = [
            "INTER-ANIMAL OCCLUSION SUMMARY",
            "=" * 40,
            "",
            f"Total Frames Analyzed: {occlusion_stats['total_frames_analyzed']}",
            f"Frames with 2+ Animals: {occlusion_stats['frames_with_multiple_animals']}",
            f"Frames with Occlusion: {frames_with_occlusion}",
            "",
            "OCCLUSION METRICS:",
            "-" * 40,
            f"Occlusion Rate: {occlusion_stats['occlusion_rate']:.1%}",
            f"Mean Severity: {occlusion_stats['mean_severity']:.3f}",
            f"Max Severity: {occlusion_stats['max_severity']:.3f}",
            "",
            "INTERPRETATION:",
            "-" * 40,
        ]

        occ_rate = occlusion_stats['occlusion_rate']
        if occ_rate < 0.05:
            summary_lines.append("Very low occlusion - animals well separated")
        elif occ_rate < 0.15:
            summary_lines.append("Low occlusion - mostly clear views")
        elif occ_rate < 0.30:
            summary_lines.append("Moderate occlusion - some blocking")
        else:
            summary_lines.append("High occlusion - dense grouping")

        # Most/least blocked animals
        if per_track:
            sorted_tracks = sorted(per_track.items(), key=lambda x: x[1], reverse=True)
            if sorted_tracks[0][1] > 0:
                summary_lines.extend([
                    "",
                    f"Most blocked: Track {sorted_tracks[0][0]} ({sorted_tracks[0][1]:.2f})",
                ])
                if len(sorted_tracks) > 1:
                    least = sorted_tracks[-1]
                    summary_lines.append(f"Least blocked: Track {least[0]} ({least[1]:.2f})")

        summary_lines.extend([
            "",
            "NOTE: Face visibility (which faces point",
            "toward camera) is in Viewpoint Analysis,",
            "not here. This shows only inter-animal",
            "blocking."
        ])

        summary_text = '\n'.join(summary_lines)
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.5))

        plt.tight_layout()

        output_path = viz_dir / "inter_animal_occlusion.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nSaved occlusion visualization to: {output_path}")

        # Save JSON
        self._save_occlusion_json(occlusion_stats, viz_dir)

        return viz_dir

    def _save_occlusion_json(self, occlusion_stats, viz_dir):
        """Save inter-animal occlusion statistics to JSON."""

        # Convert numpy types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj

        json_data = convert_types(occlusion_stats)

        output_path = viz_dir / "inter_animal_occlusion.json"
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"Saved occlusion statistics to: {output_path}")

    # ========================================================================
    # TRACKLET VIEWPOINT CHARACTERIZATION
    # ========================================================================

    def compute_tracklet_viewpoint_profiles(self):
        """Comprehensive per-tracklet viewpoint characterization."""
        print("\n" + "="*70)
        print("COMPUTING TRACKLET VIEWPOINT PROFILES")
        print("="*70)

        semantic_labels = ['front', 'back', 'left', 'right', 'top', 'bottom']
        visible_labels = ['front', 'back', 'left', 'right', 'top']  # Bottom N/A for drone footage
        tracklet_profiles = {}

        for track_id in self.labeled_tracks:
            if track_id not in self.semantic_faces:
                continue

            print(f"\nAnalyzing Track {track_id}...")

            # Initialize data structures
            visibility_matrix = {label: [] for label in semantic_labels}
            quality_matrix = {label: [] for label in semantic_labels}
            frame_quality_scores = {}

            # Analyze each frame
            for frame_name in sorted(self.semantic_faces[track_id].keys(),
                                    key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0):
                camera_params = self._load_camera_params(frame_name)
                if not camera_params:
                    continue

                # Load image for quality assessment
                img_shape = (480, 640, 3)  # Default shape
                if self.images_dir:
                    img_path = self.images_dir / f"{frame_name}.jpg"
                    if not img_path.exists():
                        img_path = self.images_dir / f"{frame_name}.png"
                    if img_path.exists():
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img_shape = img.shape

                semantic_faces = self.semantic_faces[track_id][frame_name]
                frame_qualities = {}

                for label in semantic_labels:
                    if label == 'bottom':
                        # Bottom is N/A for drone footage
                        visibility_matrix[label].append(0)
                        quality_matrix[label].append(0.0)
                        frame_qualities[label] = 0.0
                    elif label in semantic_faces:
                        # Visibility analysis
                        is_visible, visibility_score = self._calculate_face_visibility(
                            semantic_faces[label], camera_params
                        )
                        visibility_matrix[label].append(1 if is_visible else 0)

                        # Quality analysis
                        quality_score = self._calculate_face_quality(
                            semantic_faces[label], camera_params, img_shape
                        )
                        quality_matrix[label].append(quality_score)
                        frame_qualities[label] = quality_score
                    else:
                        visibility_matrix[label].append(0)
                        quality_matrix[label].append(0.0)
                        frame_qualities[label] = 0.0

                frame_quality_scores[frame_name] = frame_qualities

            # Compute viewpoint coverage vector (only for visible labels)
            total_frames = len(self.semantic_faces[track_id])
            coverage_vector = {
                label: sum(visibility_matrix[label]) / total_frames if total_frames > 0 else 0.0
                for label in semantic_labels
            }
            # Mark bottom as N/A
            coverage_vector['bottom'] = float('nan')

            # Compute viewpoint diversity index (entropy) - only for visible labels
            coverage_values = [coverage_vector[label] for label in visible_labels]
            # Add small epsilon to avoid log(0)
            coverage_values = [max(val, 1e-10) for val in coverage_values]
            coverage_sum = sum(coverage_values)
            if coverage_sum > 0:
                normalized_coverage = [val / coverage_sum for val in coverage_values]
                diversity_index = entropy(normalized_coverage)
                max_entropy = entropy([1/len(visible_labels)] * len(visible_labels))
                normalized_diversity = diversity_index / max_entropy
            else:
                diversity_index = 0.0
                normalized_diversity = 0.0

            # Compute coverage completeness score (only for visible labels)
            visible_coverage = [coverage_vector[label] for label in visible_labels]
            min_coverage = min(visible_coverage)
            coverage_threshold = 0.1  # Minimum acceptable coverage per orientation
            completeness_score = max(0.0, min(1.0, min_coverage / coverage_threshold))

            # Find optimal exemplar frames per orientation
            optimal_frames = {}
            for label in semantic_labels:
                if label == 'bottom':
                    optimal_frames[label] = {'frame': None, 'quality_score': float('nan')}
                elif any(q > 0 for q in quality_matrix[label]):
                    best_quality = max(quality_matrix[label])
                    best_frame_idx = quality_matrix[label].index(best_quality)
                    frame_names = sorted(self.semantic_faces[track_id].keys(),
                                        key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)
                    if best_frame_idx < len(frame_names):
                        optimal_frames[label] = {
                            'frame': frame_names[best_frame_idx],
                            'quality_score': best_quality
                        }
                else:
                    optimal_frames[label] = {'frame': None, 'quality_score': 0.0}

            # Compute re-identification readiness score (only considering visible labels)
            visible_avg_qualities = []
            for label in visible_labels:
                if any(q > 0 for q in quality_matrix[label]):
                    visible_avg_qualities.append(np.mean([q for q in quality_matrix[label] if q > 0]))
                else:
                    visible_avg_qualities.append(0.0)

            reid_score = (
                0.4 * normalized_diversity +      # Viewpoint diversity
                0.3 * completeness_score +        # Coverage completeness
                0.3 * np.mean(visible_avg_qualities)  # Average quality
            )

            # Store comprehensive profile
            tracklet_profiles[track_id] = {
                'coverage_vector': coverage_vector,
                'diversity_index': diversity_index,
                'normalized_diversity': normalized_diversity,
                'completeness_score': completeness_score,
                'reid_readiness_score': reid_score,
                'total_frames': total_frames,
                'visible_frames_per_orientation': {
                    label: sum(visibility_matrix[label]) if label != 'bottom' else 0
                    for label in semantic_labels
                },
                'average_quality_per_orientation': {
                    label: (np.mean([q for q in quality_matrix[label] if q > 0])
                        if label != 'bottom' and any(q > 0 for q in quality_matrix[label])
                        else (float('nan') if label == 'bottom' else 0.0))
                    for label in semantic_labels
                },
                'optimal_exemplar_frames': optimal_frames,
                'frame_quality_scores': frame_quality_scores,
                'coverage_gaps': [label for label in visible_labels
                                if coverage_vector[label] < coverage_threshold],
                'bottom_note': 'N/A - not visible in drone footage'
            }

            visible_coverage_str = [f'{coverage_vector[l]:.2f}' for l in visible_labels]
            print(f"  Coverage Vector (visible): {visible_coverage_str}")
            print(f"  (Bottom: N/A - drone footage)")
            print(f"  Diversity Index: {normalized_diversity:.3f}")
            print(f"  Completeness Score: {completeness_score:.3f}")
            print(f"  Re-ID Readiness Score: {reid_score:.3f}")
            if tracklet_profiles[track_id]['coverage_gaps']:
                print(f"  Coverage Gaps: {tracklet_profiles[track_id]['coverage_gaps']}")

        return tracklet_profiles

    def create_tracklet_viewpoint_dashboard(self):
        """Create comprehensive tracklet-focused viewpoint analysis dashboard."""
        print("\n" + "="*70)
        print("GENERATING TRACKLET VIEWPOINT CHARACTERIZATION DASHBOARD")
        print("="*70)

        viz_dir = self.annotator_output_dir / "viewpoint_analysis"
        viz_dir.mkdir(exist_ok=True)

        tracklet_profiles = self.compute_tracklet_viewpoint_profiles()

        if not tracklet_profiles:
            print("No tracklet profiles available.")
            return None

        semantic_labels = ['front', 'back', 'left', 'right', 'top', 'bottom']
        track_ids = sorted(tracklet_profiles.keys())

        # Create comprehensive dashboard
        fig = plt.figure(figsize=(24, 16))

        # 1. Per-tracklet coverage vectors (radar chart)
        ax1 = plt.subplot(2, 4, 1, projection='polar')
        angles = np.linspace(0, 2*np.pi, len(semantic_labels), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        for i, track_id in enumerate(track_ids):
            values = [tracklet_profiles[track_id]['coverage_vector'][label] for label in semantic_labels]
            # Replace NaN with 0 for plotting
            values = [0 if np.isnan(v) else v for v in values]
            values += values[:1]  # Complete the circle

            color = plt.cm.Set3(i / max(len(track_ids), 1))
            ax1.plot(angles, values, 'o-', linewidth=2, label=f'Track {track_id}', color=color)
            ax1.fill(angles, values, alpha=0.25, color=color)

        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels([label.capitalize() for label in semantic_labels])
        ax1.set_ylim(0, 1)
        ax1.set_title('Coverage Vectors per Tracklet', fontsize=14, fontweight='bold', pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax1.grid(True)

        # 2. Viewpoint diversity comparison
        ax2 = plt.subplot(2, 4, 2)
        diversity_scores = [tracklet_profiles[tid]['normalized_diversity'] for tid in track_ids]
        colors = [plt.cm.viridis(score) for score in diversity_scores]

        bars = ax2.bar(range(len(track_ids)), diversity_scores, color=colors)
        ax2.set_xlabel('Track ID', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Normalized Diversity Index', fontsize=12, fontweight='bold')
        ax2.set_title('Viewpoint Diversity per Tracklet', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(track_ids)))
        ax2.set_xticklabels([f'T{tid}' for tid in track_ids])
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)

        for bar, score in zip(bars, diversity_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 3. Coverage completeness scores
        ax3 = plt.subplot(2, 4, 3)
        completeness_scores = [tracklet_profiles[tid]['completeness_score'] for tid in track_ids]
        reid_scores = [tracklet_profiles[tid]['reid_readiness_score'] for tid in track_ids]

        x = np.arange(len(track_ids))
        width = 0.35

        bars1 = ax3.bar(x - width/2, completeness_scores, width, label='Coverage Completeness',
                       color='lightcoral', alpha=0.8)
        bars2 = ax3.bar(x + width/2, reid_scores, width, label='Re-ID Readiness',
                       color='lightblue', alpha=0.8)

        ax3.set_xlabel('Track ID', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax3.set_title('Completeness & Re-ID Readiness', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'T{tid}' for tid in track_ids])
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim(0, 1)

        # 4. Coverage gaps analysis
        ax4 = plt.subplot(2, 4, 4)
        gap_matrix = np.zeros((len(track_ids), len(semantic_labels)))

        for i, track_id in enumerate(track_ids):
            gaps = tracklet_profiles[track_id]['coverage_gaps']
            for j, label in enumerate(semantic_labels):
                gap_matrix[i, j] = 1 if label in gaps else 0

        im = ax4.imshow(gap_matrix, cmap='Reds', aspect='auto')
        ax4.set_xlabel('Semantic Face', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Track ID', fontsize=12, fontweight='bold')
        ax4.set_title('Coverage Gaps (Red = Gap)', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(semantic_labels)))
        ax4.set_xticklabels([l.capitalize() for l in semantic_labels], rotation=45)
        ax4.set_yticks(range(len(track_ids)))
        ax4.set_yticklabels([f'T{tid}' for tid in track_ids])

        # 5. Quality heatmap per orientation
        ax5 = plt.subplot(2, 4, 5)
        quality_matrix = np.array([[tracklet_profiles[tid]['average_quality_per_orientation'][label]
                                   for label in semantic_labels] for tid in track_ids])
        # Replace NaN with 0 for visualization
        quality_matrix = np.nan_to_num(quality_matrix, nan=0.0)

        im2 = ax5.imshow(quality_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax5.set_xlabel('Semantic Face', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Track ID', fontsize=12, fontweight='bold')
        ax5.set_title('Average Quality per Orientation', fontsize=14, fontweight='bold')
        ax5.set_xticks(range(len(semantic_labels)))
        ax5.set_xticklabels([l.capitalize() for l in semantic_labels], rotation=45)
        ax5.set_yticks(range(len(track_ids)))
        ax5.set_yticklabels([f'T{tid}' for tid in track_ids])

        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax5)
        cbar2.set_label('Quality Score', fontsize=10, fontweight='bold')

        # 6. Frame count distribution
        ax6 = plt.subplot(2, 4, 6)
        frame_counts = [tracklet_profiles[tid]['total_frames'] for tid in track_ids]

        bars3 = ax6.bar(range(len(track_ids)), frame_counts, color='lightgreen', alpha=0.8)
        ax6.set_xlabel('Track ID', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Total Frames', fontsize=12, fontweight='bold')
        ax6.set_title('Frame Count per Tracklet', fontsize=14, fontweight='bold')
        ax6.set_xticks(range(len(track_ids)))
        ax6.set_xticklabels([f'T{tid}' for tid in track_ids])
        ax6.grid(axis='y', alpha=0.3)

        for bar, count in zip(bars3, frame_counts):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 7. Quality distribution violin plot
        ax7 = plt.subplot(2, 4, 7)

        quality_data = []
        positions = []
        labels_plot = []

        for i, label in enumerate(semantic_labels):
            orientation_qualities = []
            for track_id in track_ids:
                avg_quality = tracklet_profiles[track_id]['average_quality_per_orientation'][label]
                if not np.isnan(avg_quality) and avg_quality > 0:
                    orientation_qualities.append(avg_quality)

            if orientation_qualities:
                quality_data.append(orientation_qualities)
                positions.append(i)
                labels_plot.append(label.capitalize())

        if quality_data:
            parts = ax7.violinplot(quality_data, positions=positions, widths=0.7,
                                  showmeans=True, showmedians=True)

            colors = [self.semantic_face_colors[semantic_labels[pos]] for pos in positions]
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)

        ax7.set_xlabel('Semantic Face', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Quality Score', fontsize=12, fontweight='bold')
        ax7.set_title('Quality Distribution per Orientation', fontsize=14, fontweight='bold')
        ax7.set_xticks(positions)
        ax7.set_xticklabels(labels_plot, rotation=45)
        ax7.grid(axis='y', alpha=0.3)
        ax7.set_ylim(0, 1)

        # 8. Summary statistics
        ax8 = plt.subplot(2, 4, 8)
        ax8.axis('off')

        summary_text = [
            "TRACKLET VIEWPOINT ANALYSIS SUMMARY",
            "="*45,
            f"Total Analyzed Tracklets: {len(tracklet_profiles)}",
            "",
            "DIVERSITY RANKING (Normalized):",
            "-"*45,
        ]

        # Sort tracklets by diversity
        sorted_tracks_diversity = sorted(track_ids,
                                       key=lambda x: tracklet_profiles[x]['normalized_diversity'],
                                       reverse=True)

        for rank, track_id in enumerate(sorted_tracks_diversity, 1):
            diversity = tracklet_profiles[track_id]['normalized_diversity']
            summary_text.append(f"{rank}. Track {track_id}: {diversity:.3f}")

        summary_text.extend([
            "",
            "RE-ID READINESS RANKING:",
            "-"*45,
        ])

        # Sort tracklets by re-ID readiness
        sorted_tracks_reid = sorted(track_ids,
                                  key=lambda x: tracklet_profiles[x]['reid_readiness_score'],
                                  reverse=True)

        for rank, track_id in enumerate(sorted_tracks_reid, 1):
            reid_score = tracklet_profiles[track_id]['reid_readiness_score']
            summary_text.append(f"{rank}. Track {track_id}: {reid_score:.3f}")

        summary_text.extend([
            "",
            "COVERAGE ISSUES:",
            "-"*45,
        ])

        for track_id in track_ids:
            gaps = tracklet_profiles[track_id]['coverage_gaps']
            if gaps:
                summary_text.append(f"Track {track_id}: Missing {', '.join(gaps)}")
            else:
                summary_text.append(f"Track {track_id}: Complete coverage")

        text_str = '\n'.join(summary_text)
        ax8.text(0.05, 0.95, text_str, transform=ax8.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.tight_layout(pad=2.0)

        # Save dashboard
        output_path = viz_dir / "tracklet_viewpoint_dashboard.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved tracklet viewpoint dashboard to: {output_path}")
        plt.close()

        # Save detailed analysis
        self._save_tracklet_analysis_json(tracklet_profiles, viz_dir)
        self._generate_optimal_frame_report(tracklet_profiles, viz_dir)

        return tracklet_profiles

    def _save_tracklet_analysis_json(self, tracklet_profiles, viz_dir):
        """Save comprehensive tracklet analysis to JSON."""
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                if np.isnan(obj):
                    return None  # JSON doesn't support NaN
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, float) and np.isnan(obj):
                return None
            else:
                return obj

        json_data = convert_numpy_types(tracklet_profiles)

        output_path = viz_dir / "tracklet_viewpoint_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"Saved detailed tracklet analysis to: {output_path}")

    def _generate_optimal_frame_report(self, tracklet_profiles, viz_dir):
        """Generate report of optimal frames for each tracklet and orientation."""
        semantic_labels = ['front', 'back', 'left', 'right', 'top', 'bottom']

        report_lines = [
            "OPTIMAL EXEMPLAR FRAMES REPORT",
            "="*60,
            "",
            "This report identifies the best frame for each orientation",
            "of each tracklet based on visibility quality, face area,",
            "central positioning, and shape preservation.",
            "",
        ]

        for track_id in sorted(tracklet_profiles.keys()):
            profile = tracklet_profiles[track_id]
            report_lines.extend([
                f"TRACK {track_id}:",
                "-"*20,
                f"Diversity Index: {profile['normalized_diversity']:.3f}",
                f"Re-ID Readiness: {profile['reid_readiness_score']:.3f}",
                f"Total Frames: {profile['total_frames']}",
                "",
                "Optimal Frames by Orientation:",
            ])

            for label in semantic_labels:
                exemplar = profile['optimal_exemplar_frames'][label]
                if exemplar['frame'] is not None:
                    quality = exemplar['quality_score']
                    if not np.isnan(quality):
                        report_lines.append(
                            f"  {label.capitalize():<8}: Frame {exemplar['frame']} "
                            f"(Quality: {quality:.3f})"
                        )
                    else:
                        report_lines.append(f"  {label.capitalize():<8}: N/A (drone footage)")
                else:
                    report_lines.append(f"  {label.capitalize():<8}: No good frames available")

            if profile['coverage_gaps']:
                report_lines.extend([
                    "",
                    f"Coverage Gaps: {', '.join(profile['coverage_gaps'])}",
                ])

            report_lines.extend(["", ""])

        # Summary statistics
        report_lines.extend([
            "DATASET SUMMARY:",
            "="*30,
            "",
            f"Total Tracklets: {len(tracklet_profiles)}",
            "",
            "Average Scores:",
        ])

        avg_diversity = np.mean([p['normalized_diversity'] for p in tracklet_profiles.values()])
        avg_reid = np.mean([p['reid_readiness_score'] for p in tracklet_profiles.values()])
        avg_completeness = np.mean([p['completeness_score'] for p in tracklet_profiles.values()])

        report_lines.extend([
            f"  Diversity Index: {avg_diversity:.3f}",
            f"  Re-ID Readiness: {avg_reid:.3f}",
            f"  Coverage Completeness: {avg_completeness:.3f}",
            "",
        ])

        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS:",
            "="*20,
            "",
        ])

        # Find tracklets with low diversity
        low_diversity = [tid for tid, p in tracklet_profiles.items()
                        if p['normalized_diversity'] < 0.5]
        if low_diversity:
            report_lines.append(f"Collect more diverse viewpoints for tracks: {', '.join(map(str, low_diversity))}")

        # Find common coverage gaps
        all_gaps = []
        for p in tracklet_profiles.values():
            all_gaps.extend(p['coverage_gaps'])

        if all_gaps:
            gap_counts = Counter(all_gaps)
            common_gaps = [gap for gap, count in gap_counts.items() if count >= len(tracklet_profiles) // 2]
            if common_gaps:
                report_lines.append(f"Systematic coverage gaps detected: {', '.join(common_gaps)}")

        # Save report
        output_path = viz_dir / "optimal_frames_report.txt"
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"Saved optimal frames report to: {output_path}")

    def create_temporal_viewpoint_visualization(self, tracklet_profiles=None):
        """Create temporal viewpoint distribution visualizations."""
        print("\n" + "="*70)
        print("GENERATING TEMPORAL VIEWPOINT ANALYSIS")
        print("="*70)

        viz_dir = self.annotator_output_dir / "viewpoint_analysis"
        viz_dir.mkdir(exist_ok=True)

        if tracklet_profiles is None:
            tracklet_profiles = self.compute_tracklet_viewpoint_profiles()

        if not tracklet_profiles:
            print("No tracklet profiles available.")
            return

        track_ids = sorted(tracklet_profiles.keys())

        # Create individual temporal analysis for each tracklet
        for track_id in track_ids:
            self._create_single_tracklet_temporal_viz(track_id, tracklet_profiles[track_id], viz_dir)

        # Create a compact summary overview comparing all tracklets
        if len(track_ids) > 1:
            self._create_temporal_summary_overview(tracklet_profiles, viz_dir)

        print(f"\nGenerated {len(track_ids)} individual temporal visualizations")
        return viz_dir

    def _create_single_tracklet_temporal_viz(self, track_id, profile, viz_dir):
        """Create comprehensive temporal visualization for a single tracklet."""

        if track_id not in self.semantic_faces:
            return

        semantic_labels = ['front', 'back', 'left', 'right', 'top', 'bottom']
        visible_labels = ['front', 'back', 'left', 'right', 'top']

        # Color map - bottom grayed out
        face_cmap = {
            'front': '#E74C3C', 'back': '#2ECC71', 'left': '#3498DB',
            'right': '#F39C12', 'top': '#9B59B6', 'bottom': '#CCCCCC',
        }

        frame_quality_scores = profile['frame_quality_scores']
        sorted_frames = sorted(frame_quality_scores.keys(),
                            key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)
        n_frames = len(sorted_frames)

        if n_frames == 0:
            return

        frame_numbers = [int(re.findall(r'\d+', f)[0]) if re.findall(r'\d+', f) else i
                        for i, f in enumerate(sorted_frames)]

        # Build quality matrix
        quality_matrix = np.zeros((len(semantic_labels), n_frames))

        for frame_idx, frame_name in enumerate(sorted_frames):
            camera_params = self._load_camera_params(frame_name)
            if camera_params is None:
                continue

            semantic_faces_frame = self.semantic_faces[track_id].get(frame_name, {})

            for label_idx, label in enumerate(semantic_labels):
                if label == 'bottom':
                    quality_matrix[label_idx, frame_idx] = np.nan
                elif label in semantic_faces_frame:
                    quality_matrix[label_idx, frame_idx] = frame_quality_scores[frame_name].get(label, 0)

        # Create figure
        fig = plt.figure(figsize=(18, 14))

        # Title
        fig.suptitle(f'Track {track_id} - Temporal Viewpoint Analysis\n({n_frames} frames)',
                    fontsize=18, fontweight='bold', y=0.98)

        # Layout: 3 rows, with different configurations
        gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 0.5, 1],
                            hspace=0.35, wspace=0.3,
                            left=0.08, right=0.92, top=0.90, bottom=0.08)

        # ===== Panel 1: Quality Heatmap (top, spans 2 columns) =====
        ax_heatmap = fig.add_subplot(gs[0, :2])

        quality_matrix_masked = np.ma.array(quality_matrix, mask=np.isnan(quality_matrix))
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_bad(color='#E0E0E0')

        im = ax_heatmap.imshow(quality_matrix_masked, aspect='auto', cmap=cmap,
                            vmin=0, vmax=1, interpolation='nearest')

        # Add hatching for bottom row
        bottom_idx = semantic_labels.index('bottom')
        ax_heatmap.add_patch(plt.Rectangle((-0.5, bottom_idx - 0.5), n_frames, 1,
                                        fill=True, facecolor='#E0E0E0',
                                        edgecolor='gray', linewidth=1,
                                        hatch='///', alpha=0.7, zorder=2))

        # Y-axis labels
        y_labels = [f'{l.capitalize()}' if l != 'bottom' else 'Bottom (N/A)' for l in semantic_labels]
        ax_heatmap.set_yticks(range(len(semantic_labels)))
        ax_heatmap.set_yticklabels(y_labels, fontsize=11)
        ytick_labels = ax_heatmap.get_yticklabels()
        ytick_labels[-1].set_color('gray')
        ytick_labels[-1].set_fontstyle('italic')

        # X-axis
        tick_step = max(1, n_frames // 12)
        tick_positions = list(range(0, n_frames, tick_step))
        ax_heatmap.set_xticks(tick_positions)
        ax_heatmap.set_xticklabels([str(frame_numbers[i]) for i in tick_positions], fontsize=9)
        ax_heatmap.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
        ax_heatmap.set_ylabel('Face', fontsize=12, fontweight='bold')
        ax_heatmap.set_title('Face Quality Over Time', fontsize=14, fontweight='bold', pad=10)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8, pad=0.02)
        cbar.set_label('Quality Score', fontsize=11)

        # ===== Panel 2: Stats Summary (top right) =====
        ax_stats = fig.add_subplot(gs[0, 2])
        ax_stats.axis('off')

        stats_text = [
            f"TRACK {track_id} STATISTICS",
            "-" * 25,
            f"Total Frames: {n_frames}",
            f"",
            f"Diversity Index: {profile['normalized_diversity']:.3f}",
            f"Completeness: {profile['completeness_score']:.3f}",
            f"Re-ID Readiness: {profile['reid_readiness_score']:.3f}",
            f"",
            "Avg Quality per Face:",
        ]

        for label in visible_labels:
            avg_q = profile['average_quality_per_orientation'][label]
            if not np.isnan(avg_q):
                stats_text.append(f"  {label.capitalize():<8}: {avg_q:.3f}")
            else:
                stats_text.append(f"  {label.capitalize():<8}: N/A")

        stats_text.append(f"  {'Bottom':<8}: N/A (drone)")

        if profile['coverage_gaps']:
            stats_text.extend(["", f"Coverage Gaps:", f"  {', '.join(profile['coverage_gaps'])}"])

        ax_stats.text(0.1, 0.95, '\n'.join(stats_text), transform=ax_stats.transAxes,
                    fontsize=10, fontfamily='monospace', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))

        # ===== Panel 3: Dominant Viewpoint Timeline (middle) =====
        ax_dominant = fig.add_subplot(gs[1, :])

        # Calculate dominant views
        dominant_views = []
        dominant_colors = []

        for frame_idx in range(n_frames):
            qualities = {label: quality_matrix[semantic_labels.index(label), frame_idx]
                        for label in visible_labels}
            valid_qualities = {k: v for k, v in qualities.items() if not np.isnan(v) and v > 0}

            if valid_qualities:
                dominant_label = max(valid_qualities, key=valid_qualities.get)
                dominant_views.append(dominant_label)
                dominant_colors.append(face_cmap[dominant_label])
            else:
                dominant_views.append('none')
                dominant_colors.append('#AAAAAA')

        # Draw ribbon
        for frame_idx in range(n_frames):
            ax_dominant.barh(0, 1, left=frame_idx, color=dominant_colors[frame_idx],
                            edgecolor='none', linewidth=0)

        # Add transition markers
        transitions = []
        for i in range(1, len(dominant_views)):
            if dominant_views[i] != dominant_views[i-1] and dominant_views[i] != 'none':
                transitions.append(i)
                ax_dominant.axvline(x=i, color='white', linestyle='-', linewidth=2, alpha=0.8)

        ax_dominant.set_xlim(0, n_frames)
        ax_dominant.set_ylim(-0.5, 0.5)
        ax_dominant.set_yticks([])
        ax_dominant.set_xlabel('Frame Index', fontsize=11)
        ax_dominant.set_title(f'Dominant Viewpoint Timeline ({len(transitions)} transitions)',
                            fontsize=13, fontweight='bold')

        # Add legend below timeline
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=face_cmap[l], edgecolor='black',
                                        label=l.capitalize()) for l in visible_labels]
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='#CCCCCC', edgecolor='gray',
                                            hatch='///', label='Bottom (N/A)'))
        ax_dominant.legend(handles=legend_elements, loc='upper center',
                        bbox_to_anchor=(0.5, -0.3), ncol=6, fontsize=9)

        # ===== Panel 4: Line Plot (bottom left) =====
        ax_lines = fig.add_subplot(gs[2, :2])

        for label in visible_labels:
            label_idx = semantic_labels.index(label)
            quality_values = quality_matrix[label_idx]
            ax_lines.plot(frame_numbers, quality_values, color=face_cmap[label],
                        linewidth=2, label=label.capitalize(), marker='o',
                        markersize=2, alpha=0.8)

        ax_lines.axhspan(-0.02, 0.02, color='#E0E0E0', alpha=0.5, zorder=0)
        ax_lines.text(frame_numbers[0], 0, ' Bottom N/A', fontsize=8, color='gray',
                    fontstyle='italic', va='center')

        ax_lines.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
        ax_lines.set_ylabel('Quality Score', fontsize=12, fontweight='bold')
        ax_lines.set_title('Quality Trends by Face', fontsize=13, fontweight='bold')
        ax_lines.legend(loc='upper right', ncol=3, fontsize=9)
        ax_lines.grid(True, alpha=0.3)
        ax_lines.set_ylim(-0.05, 1.05)
        ax_lines.set_xlim(frame_numbers[0], frame_numbers[-1])

        # ===== Panel 5: Pie Chart + Transitions (bottom right) =====
        ax_pie = fig.add_subplot(gs[2, 2])

        # Pie chart of viewpoint distribution
        view_counts = {label: sum(1 for v in dominant_views if v == label) for label in visible_labels}
        view_counts = {k: v for k, v in view_counts.items() if v > 0}

        if view_counts:
            colors_pie = [face_cmap[l] for l in view_counts.keys()]
            wedges, texts, autotexts = ax_pie.pie(
                view_counts.values(),
                labels=[l.capitalize() for l in view_counts.keys()],
                colors=colors_pie,
                autopct='%1.1f%%',
                pctdistance=0.75,
                startangle=90,
                textprops={'fontsize': 9}
            )
            ax_pie.set_title('Viewpoint Distribution', fontsize=12, fontweight='bold')
        else:
            ax_pie.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
            ax_pie.set_title('Viewpoint Distribution', fontsize=12, fontweight='bold')

        # Save
        output_path = viz_dir / f"temporal_track_{track_id}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"  Saved: {output_path.name}")

    def _create_temporal_summary_overview(self, tracklet_profiles, viz_dir):
        """Create a compact summary comparing temporal patterns across all tracklets."""

        semantic_labels = ['front', 'back', 'left', 'right', 'top', 'bottom']
        visible_labels = ['front', 'back', 'left', 'right', 'top']
        track_ids = sorted(tracklet_profiles.keys())
        n_tracks = len(track_ids)

        face_cmap = {
            'front': '#E74C3C', 'back': '#2ECC71', 'left': '#3498DB',
            'right': '#F39C12', 'top': '#9B59B6', 'bottom': '#CCCCCC',
        }

        fig = plt.figure(figsize=(16, 4 + 2 * n_tracks))

        # Create grid
        gs = fig.add_gridspec(n_tracks + 1, 4, height_ratios=[0.8] + [1] * n_tracks,
                            hspace=0.4, wspace=0.3,
                            left=0.08, right=0.92, top=0.92, bottom=0.08)

        # ===== Header Row: Comparison Charts =====

        # Diversity comparison
        ax_diversity = fig.add_subplot(gs[0, 0])
        diversity_scores = [tracklet_profiles[tid]['normalized_diversity'] for tid in track_ids]
        colors = [plt.cm.viridis(score) for score in diversity_scores]
        bars = ax_diversity.bar(range(n_tracks), diversity_scores, color=colors)
        ax_diversity.set_xticks(range(n_tracks))
        ax_diversity.set_xticklabels([f'T{tid}' for tid in track_ids], fontsize=9)
        ax_diversity.set_ylabel('Score', fontsize=10)
        ax_diversity.set_title('Diversity', fontsize=11, fontweight='bold')
        ax_diversity.set_ylim(0, 1)
        ax_diversity.grid(axis='y', alpha=0.3)

        # Re-ID readiness comparison
        ax_reid = fig.add_subplot(gs[0, 1])
        reid_scores = [tracklet_profiles[tid]['reid_readiness_score'] for tid in track_ids]
        colors_reid = [plt.cm.plasma(score) for score in reid_scores]
        ax_reid.bar(range(n_tracks), reid_scores, color=colors_reid)
        ax_reid.set_xticks(range(n_tracks))
        ax_reid.set_xticklabels([f'T{tid}' for tid in track_ids], fontsize=9)
        ax_reid.set_ylabel('Score', fontsize=10)
        ax_reid.set_title('Re-ID Readiness', fontsize=11, fontweight='bold')
        ax_reid.set_ylim(0, 1)
        ax_reid.grid(axis='y', alpha=0.3)

        # Frame count comparison
        ax_frames = fig.add_subplot(gs[0, 2])
        frame_counts = [tracklet_profiles[tid]['total_frames'] for tid in track_ids]
        ax_frames.bar(range(n_tracks), frame_counts, color='lightgreen', edgecolor='darkgreen')
        ax_frames.set_xticks(range(n_tracks))
        ax_frames.set_xticklabels([f'T{tid}' for tid in track_ids], fontsize=9)
        ax_frames.set_ylabel('Frames', fontsize=10)
        ax_frames.set_title('Frame Count', fontsize=11, fontweight='bold')
        ax_frames.grid(axis='y', alpha=0.3)

        # Legend
        ax_legend = fig.add_subplot(gs[0, 3])
        ax_legend.axis('off')

        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=face_cmap[l], edgecolor='black',
                                        label=l.capitalize()) for l in visible_labels]
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='#CCCCCC', edgecolor='gray',
                                            hatch='///', label='Bottom (N/A)'))
        ax_legend.legend(handles=legend_elements, loc='center', ncol=2, fontsize=9,
                        title='Viewpoints', title_fontsize=10)

        # ===== Per-Tracklet Dominant View Ribbons =====
        for track_idx, track_id in enumerate(track_ids):
            ax_ribbon = fig.add_subplot(gs[track_idx + 1, :])

            profile = tracklet_profiles[track_id]
            frame_quality_scores = profile['frame_quality_scores']
            sorted_frames = sorted(frame_quality_scores.keys(),
                                key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)
            n_frames = len(sorted_frames)

            if n_frames == 0:
                ax_ribbon.text(0.5, 0.5, f'Track {track_id}: No frames',
                            ha='center', va='center', fontsize=12)
                ax_ribbon.axis('off')
                continue

            # Build quality for this track
            quality_matrix = np.zeros((len(semantic_labels), n_frames))
            for frame_idx, frame_name in enumerate(sorted_frames):
                for label_idx, label in enumerate(semantic_labels):
                    if label == 'bottom':
                        quality_matrix[label_idx, frame_idx] = np.nan
                    else:
                        quality_matrix[label_idx, frame_idx] = frame_quality_scores[frame_name].get(label, 0)

            # Dominant views
            dominant_views = []
            for frame_idx in range(n_frames):
                qualities = {label: quality_matrix[semantic_labels.index(label), frame_idx]
                            for label in visible_labels}
                valid_qualities = {k: v for k, v in qualities.items() if not np.isnan(v) and v > 0}

                if valid_qualities:
                    dominant_views.append(max(valid_qualities, key=valid_qualities.get))
                else:
                    dominant_views.append('none')

            # Draw ribbon
            for frame_idx in range(n_frames):
                color = face_cmap.get(dominant_views[frame_idx], '#AAAAAA')
                ax_ribbon.barh(0, 1, left=frame_idx, color=color, edgecolor='none')

            # Transitions
            transitions = sum(1 for i in range(1, len(dominant_views))
                            if dominant_views[i] != dominant_views[i-1] and dominant_views[i] != 'none')

            ax_ribbon.set_xlim(0, n_frames)
            ax_ribbon.set_ylim(-0.5, 0.5)
            ax_ribbon.set_yticks([])
            ax_ribbon.set_xlabel('Frame Index', fontsize=9)

            # Track info on the left
            ax_ribbon.set_ylabel(f'T{track_id}', fontsize=12, fontweight='bold', rotation=0,
                                labelpad=30, va='center')

            # Stats annotation on right
            stats_str = f'n={n_frames} | div={profile["normalized_diversity"]:.2f} | trans={transitions}'
            ax_ribbon.text(1.01, 0.5, stats_str, transform=ax_ribbon.transAxes,
                        fontsize=9, va='center', fontfamily='monospace')

        fig.suptitle('Temporal Viewpoint Summary - All Tracklets\n(Bottom face N/A - drone footage)',
                    fontsize=14, fontweight='bold', y=0.98)

        output_path = viz_dir / "temporal_summary_overview.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"  Saved summary overview: {output_path.name}")

    def print_analysis_summary(self):
        """Print concise analysis summary."""
        tracklet_profiles = self.compute_tracklet_viewpoint_profiles()

        if not tracklet_profiles:
            print("No tracklet profiles available.")
            return

        print("\n" + "="*70)
        print("VIEWPOINT ANALYSIS SUMMARY")
        print("="*70)

        semantic_labels = ['front', 'back', 'left', 'right', 'top', 'bottom']
        visible_labels = ['front', 'back', 'left', 'right', 'top']

        print(f"\nANALYZED TRACKLETS: {len(tracklet_profiles)}")

        for track_id in sorted(tracklet_profiles.keys()):
            profile = tracklet_profiles[track_id]
            print(f"\n  TRACK {track_id}:")
            print(f"    Frames: {profile['total_frames']}")
            coverage_values = [f'{profile["coverage_vector"][l]:.2f}' for l in visible_labels]
            print(f"    Coverage: {coverage_values}")
            print(f"    Diversity Index: {profile['normalized_diversity']:.3f}")
            print(f"    Re-ID Readiness: {profile['reid_readiness_score']:.3f}")
            if profile['coverage_gaps']:
                print(f"    Coverage Gaps: {', '.join(profile['coverage_gaps'])}")

        # Dataset-level statistics
        avg_diversity = np.mean([p['normalized_diversity'] for p in tracklet_profiles.values()])
        avg_reid = np.mean([p['reid_readiness_score'] for p in tracklet_profiles.values()])

        print(f"\nDATASET AVERAGES:")
        print(f"  Diversity Index: {avg_diversity:.3f}")
        print(f"  Re-ID Readiness: {avg_reid:.3f}")

        print("\n" + "="*70)


def main():
    """Command-line interface for viewpoint and occlusion analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Viewpoint and occlusion analysis for WildLIFT annotated wildlife tracklets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python viewpoint_analyzer_v2.py --annotator_output results/zebra/scene1/corrected/

    # With original images for enhanced visualization
    python viewpoint_analyzer_v2.py \\
        --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/

    # Include occlusion analysis
    python viewpoint_analyzer_v2.py \\
        --annotator_output results/zebra/scene1/corrected/ \\
        --compute_occlusion

    # Full analysis with all features
    python viewpoint_analyzer_v2.py \\
        --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/ \\
        --compute_occlusion --validate
        """
    )

    parser.add_argument(
        "--annotator_output",
        required=True,
        help="Path to annotator tool output directory containing "
             "bounding_boxes/ and corrected_labels/"
    )

    parser.add_argument(
        "--images_dir",
        default=None,
        help="Path to original images (optional, for visualization)"
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for analysis results. "
             "Default: {annotator_output}/viewpoint_analysis/"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run semantic consistency validation before analysis"
    )

    parser.add_argument(
        "--skip_visualizations",
        action="store_true",
        help="Skip generating visualization images"
    )

    parser.add_argument(
        "--compute_occlusion",
        action="store_true",
        help="Compute inter-animal occlusion statistics"
    )

    args = parser.parse_args()

    # Initialize analyzer
    print("=" * 70)
    print("WildLIFT VIEWPOINT ANALYZER v2")
    print("=" * 70)

    try:
        analyzer = ViewpointAnalyzer(
            annotator_output_dir=args.annotator_output,
            images_dir=args.images_dir
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nEnsure you have run the annotator tool and labeled semantic faces.")
        return 1

    # Validation (optional)
    if args.validate:
        print("\n" + "-" * 40)
        print("Validating semantic consistency...")
        issues = analyzer.validate_semantic_consistency()

        if issues:
            print(f"Found {len(issues)} consistency issues:")
            for issue in issues[:10]:  # Show first 10
                print(f"  Track {issue['track']}, {issue['face']}: "
                      f"{issue['issue_type']} (severity: {issue['severity']:.2f})")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
        else:
            print("No consistency issues found.")

    # Generate visualizations
    if not args.skip_visualizations:
        print("\n" + "-" * 40)
        print("Generating visualizations...")

        tracklet_profiles = analyzer.create_tracklet_viewpoint_dashboard()
        if tracklet_profiles:
            analyzer.create_temporal_viewpoint_visualization(tracklet_profiles)
    else:
        print("\n" + "-" * 40)
        print("Computing viewpoint profiles (skipping visualizations)...")
        analyzer.print_analysis_summary()

    # Occlusion analysis (optional)
    if args.compute_occlusion:
        print("\n" + "-" * 40)
        print("Computing occlusion statistics...")
        occlusion_stats = analyzer.compute_occlusion_statistics()
        analyzer.create_occlusion_visualizations(occlusion_stats)

    output_dir = args.output_dir or (args.annotator_output + "/viewpoint_analysis")

    print("\n" + "=" * 70)
    print(f"Analysis complete. Results saved to: {output_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
