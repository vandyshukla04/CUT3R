#!/usr/bin/env python3
"""
Viewpoint Analyzer v4 for WildLIFT Pipeline

Objective reporting engine for viewpoint characterization with high-density
"Filmstrip" PDF reports providing temporally diverse visual evidence.

Key Changes from v3:
    - Removed subjective "Re-ID Readiness" and "Data Collection QC" scores
    - Implements Temporal Non-Maximum Suppression (NMS) for diverse exemplar selection
    - Generates Portrait/Filmstrip PDF layout (5 crops per orientation)
    - All orientations treated equally (no biased weights)
    - Configurable via module-level CONFIG dictionary

This module provides:
    1. VIEWPOINT ANALYSIS (visibility/coverage)
    2. INTER-ANIMAL OCCLUSION ANALYSIS (physical blocking)
    3. FILMSTRIP REPORTING (temporally diverse visual evidence)

Usage:
    # Basic analysis with PDF report
    python viewpoint_analyzer_v4.py --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/

    # With occlusion analysis
    python viewpoint_analyzer_v4.py --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/ --compute_occlusion

    # Specify mask directory explicitly
    python viewpoint_analyzer_v4.py --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/ --mask_dir data/zebra/scene1/grounded-sam/
"""

import os
import json
import numpy as np
import cv2
import glob
import argparse
from pathlib import Path
import re
import colorsys
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
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

# Optional: pycocotools for mask decoding
try:
    from pycocotools import mask as mask_utils
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("Warning: pycocotools not available. Mask extraction will be limited.")


# =============================================================================
# GLOBAL CONFIGURATION (NEW IN V4)
# =============================================================================

CONFIG = {
    # Reporting Settings
    'CROPS_PER_FACE': 5,            # Target number of mask crops per orientation
    'PDF_LAYOUT': 'portrait',       # Report alignment
    'PDF_FIGURE_SIZE': (8.5, 11),   # Standard Letter size
    'CROP_PADDING': 30,             # Pixels around mask bbox
    'CROP_BG_COLOR': 'white',       # 'white' or 'transparent'

    # Temporal Diversity (NMS)
    'NMS_FRAME_GAP': 30,            # Minimum frames between selected crops to ensure diversity

    # Grading Thresholds (Strictly Geometric)
    'THRESHOLDS': {
        'coverage_good': 0.15,      # >15% visibility required for "Good" rating
        'coverage_fair': 0.08,      # >8% for "Fair" rating
        'occlusion_severe': 0.30,   # >30% overlap considered severe
        'diversity_good': 0.60,     # >0.6 normalized diversity for "Good"
        'diversity_fair': 0.40,     # >0.4 for "Fair"
    },

    # Semantic face labels (visible faces for drone footage)
    'VISIBLE_FACES': ['front', 'back', 'left', 'right', 'top'],
    'ALL_FACES': ['front', 'back', 'left', 'right', 'top', 'bottom'],
}


# =============================================================================
# DATA CLASSES FOR GRADING (SIMPLIFIED IN V4)
# =============================================================================

@dataclass
class GradeResult:
    """Result of grading a metric (strictly geometric)."""
    letter: str  # A, B, C, F
    score: float  # 0-1 normalized score
    label: str  # "Excellent", "Good", "Fair", "Poor"
    description: str  # Plain-English explanation
    color: str  # For visualization


@dataclass
class TrackletGrades:
    """Geometric grades for a single tracklet (v4: no subjective scores)."""
    overall: GradeResult
    coverage: GradeResult
    diversity: GradeResult
    quality: GradeResult


@dataclass
class ExemplarFrame:
    """Information about a single exemplar frame."""
    frame: str
    quality_score: float
    frame_idx: int  # For temporal ordering


@dataclass
class TrackletProfile:
    """Complete profile for a tracklet (v4 schema)."""
    track_id: int
    total_frames: int
    coverage_vector: Dict[str, float]
    diversity_index: float
    normalized_diversity: float
    visible_frames_per_orientation: Dict[str, int]
    average_quality_per_orientation: Dict[str, float]
    exemplars: Dict[str, List[ExemplarFrame]]  # NEW: List of diverse exemplars
    coverage_gaps: List[str]
    frame_quality_scores: Dict[str, Dict[str, float]]


# =============================================================================
# VIEWPOINT ANALYZER CLASS
# =============================================================================

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
            images_dir: Optional path to original images for visualization.
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

        print(f"Viewpoint Analyzer v4 initialized:")
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
        """Load bounding boxes from annotator tool output."""
        bbox_dir = self.annotator_output_dir / "bounding_boxes"
        all_bbox_data = {}

        for bbox_file in sorted(bbox_dir.glob("*.json")):
            frame_name = bbox_file.stem

            with open(bbox_file, 'r') as f:
                frame_bboxes = json.load(f)

            for bbox in frame_bboxes:
                track_id = bbox.get('track_id')

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
        """Load semantic face labels from annotator tool output."""
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
                frame_key = str(frame_name)

                if frame_key not in self.all_bbox_data[track_id]:
                    print(f"  Warning: Labels for track {track_id} frame {frame_key} "
                          "but no bbox data")
                    continue

                bbox_data = self.all_bbox_data[track_id][frame_key]
                all_faces = self.get_all_faces_from_bbox(bbox_data)

                semantic_faces[track_id][frame_key] = {}

                for semantic_label, face_index in label_to_index.items():
                    face_key = f'f{face_index}'

                    if face_key in all_faces:
                        semantic_faces[track_id][frame_key][semantic_label] = all_faces[face_key]
                    else:
                        print(f"  Warning: Face index {face_index} invalid for "
                              f"track {track_id} frame {frame_key}")

                self._infer_opposite_faces(semantic_faces[track_id][frame_key], all_faces)

        total_labels = sum(len(frames) for frames in semantic_faces.values())
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
        """Load camera parameters for a specific frame."""
        camera_file = self.annotator_output_dir / "camera" / f"{frame_name}.npz"

        if not camera_file.exists():
            results_dir = self.annotator_output_dir.parent
            camera_file = results_dir / "camera" / f"{frame_name}.npz"

        if not camera_file.exists():
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
            [-l/2, -w/2, -h/2],
            [+l/2, -w/2, -h/2],
            [+l/2, +w/2, -h/2],
            [-l/2, +w/2, -h/2],
            [-l/2, -w/2, +h/2],
            [+l/2, -w/2, +h/2],
            [+l/2, +w/2, +h/2],
            [-l/2, +w/2, +h/2],
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
        """Get all 6 faces from bbox data."""
        corners = self.get_bbox_corners(
            bbox_data['center'],
            bbox_data['dimensions'],
            bbox_data['rotation_matrix']
        )
        box_center = np.mean(corners, axis=0)

        face_indices = {
            'f0': [0, 1, 5, 4],
            'f1': [2, 3, 7, 6],
            'f2': [0, 3, 7, 4],
            'f3': [1, 2, 6, 5],
            'f4': [4, 5, 6, 7],
            'f5': [0, 1, 2, 3],
        }

        faces = {}
        for face_id, indices in face_indices.items():
            faces[face_id] = self.compute_face_from_corners(corners, indices, box_center)

        return faces

    def _infer_opposite_faces(self, semantic_assignments, all_faces):
        """Infer opposite faces based on normal similarity."""
        opposites = {'front': 'back', 'left': 'right', 'top': 'bottom'}

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

                if opposite_label in semantic_assignments:
                    continue

                best_match = None
                best_score = 999

                for face_id, test_face in unassigned_faces.items():
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
            t = camera_params['t']
            camera_pos = t

            face_to_camera = camera_pos - face_data['center']
            face_to_camera = face_to_camera / np.linalg.norm(face_to_camera)

            visibility_score = np.dot(face_data['normal'], face_to_camera)

            return visibility_score > 0, visibility_score
        except:
            return False, 0.0

    def _calculate_face_quality(self, face_data, camera_params, img_shape):
        """Calculate comprehensive face quality score for frame ranking."""
        try:
            is_visible, visibility_score = self._calculate_face_visibility(face_data, camera_params)
            if not is_visible:
                return 0.0

            corners_2d = self._project_face_to_2d(face_data, camera_params, img_shape)
            if corners_2d is None:
                return 0.0

            face_area_2d = cv2.contourArea(corners_2d)
            max_possible_area = img_shape[0] * img_shape[1]
            area_score = min(face_area_2d / max_possible_area, 1.0)

            img_center = np.array([img_shape[1]/2, img_shape[0]/2])
            face_center_2d = np.mean(corners_2d, axis=0)
            max_distance = np.sqrt(img_shape[0]**2 + img_shape[1]**2) / 2
            distance_score = 1.0 - np.linalg.norm(face_center_2d - img_center) / max_distance

            x_span = np.max(corners_2d[:, 0]) - np.min(corners_2d[:, 0])
            y_span = np.max(corners_2d[:, 1]) - np.min(corners_2d[:, 1])
            if min(x_span, y_span) > 0:
                aspect_ratio = min(x_span, y_span) / max(x_span, y_span)
            else:
                aspect_ratio = 0.0

            # V4: Equal weights for all components (no bias toward certain faces)
            quality_score = (
                0.25 * abs(visibility_score) +
                0.25 * area_score +
                0.25 * distance_score +
                0.25 * aspect_ratio
            )

            return quality_score
        except:
            return 0.0

    # ========================================================================
    # TEMPORAL NMS FOR DIVERSE EXEMPLAR SELECTION (NEW IN V4)
    # ========================================================================

    def _select_diverse_exemplars(self, quality_data: List[Tuple[int, str, float]],
                                   n_count: int = None) -> List[ExemplarFrame]:
        """
        Select temporally diverse exemplar frames using Non-Maximum Suppression.

        Args:
            quality_data: List of tuples (frame_idx, frame_name, quality_score)
            n_count: Number of exemplars to select (default from CONFIG)

        Returns:
            List of ExemplarFrame objects, sorted chronologically
        """
        if n_count is None:
            n_count = CONFIG['CROPS_PER_FACE']

        nms_gap = CONFIG['NMS_FRAME_GAP']

        if not quality_data:
            return []

        # Sort by quality score descending
        candidates = sorted(quality_data, key=lambda x: x[2], reverse=True)

        selected = []

        while candidates and len(selected) < n_count:
            # Pick the best remaining candidate
            best = candidates[0]
            best_idx, best_frame, best_quality = best

            # Add to selected
            selected.append(ExemplarFrame(
                frame=best_frame,
                quality_score=best_quality,
                frame_idx=best_idx
            ))

            # Remove best from candidates
            candidates = candidates[1:]

            # Suppress nearby frames (within NMS_FRAME_GAP)
            candidates = [
                c for c in candidates
                if abs(c[0] - best_idx) >= nms_gap
            ]

        # Sort selected by frame index (chronological order)
        selected.sort(key=lambda x: x.frame_idx)

        return selected

    # ========================================================================
    # OCCLUSION ANALYSIS METHODS
    # ========================================================================

    def _get_bboxes_for_frame(self, frame_name):
        """Get all bounding boxes present in a specific frame."""
        frame_bboxes = []

        for track_id, track_data in self.all_bbox_data.items():
            if frame_name in track_data:
                bbox_data = track_data[frame_name].copy()
                bbox_data['track_id'] = track_id
                frame_bboxes.append(bbox_data)

        return frame_bboxes

    def _project_bbox_to_2d(self, bbox_data, camera_params):
        """Project 3D OBB corners to 2D image coordinates."""
        corners_3d = self.get_bbox_corners(
            bbox_data['center'],
            bbox_data['dimensions'],
            bbox_data['rotation_matrix']
        )

        K = camera_params['K']
        R = camera_params['R']
        t = camera_params['t']

        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R
        camera_pose[:3, 3] = t

        corners_h = np.concatenate([corners_3d, np.ones((8, 1))], axis=1)
        corners_cam = (np.linalg.inv(camera_pose) @ corners_h.T).T[:, :3]

        if np.any(corners_cam[:, 2] <= 0):
            return None

        corners_2d_h = (K @ corners_cam.T).T
        corners_2d = corners_2d_h[:, :2] / corners_2d_h[:, 2:3]

        return corners_2d

    def _compute_2d_overlap(self, corners_a, corners_b):
        """Compute fraction of hull_b's area overlapped by hull_a."""
        if not SCIPY_AVAILABLE or not SHAPELY_AVAILABLE:
            return 0.0

        try:
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

        except Exception:
            return 0.0

    def compute_inter_animal_occlusion(self, frame_name):
        """Compute pairwise occlusion matrix for all animals in a frame."""
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

        camera_pos = camera_params['t']

        occlusion_matrix = np.zeros((n, n))
        track_ids = [b['track_id'] for b in frame_bboxes]

        projections = []
        distances = []

        for bbox in frame_bboxes:
            corners_2d = self._project_bbox_to_2d(bbox, camera_params)
            projections.append(corners_2d)

            dist = np.linalg.norm(bbox['center'] - camera_pos)
            distances.append(dist)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                if projections[i] is None or projections[j] is None:
                    continue

                if distances[i] >= distances[j]:
                    continue

                overlap = self._compute_2d_overlap(projections[i], projections[j])
                occlusion_matrix[i, j] = overlap

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
        """Compute inter-animal occlusion statistics across all frames."""
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

            occ_result = self.compute_inter_animal_occlusion(frame_name)

            if occ_result is None:
                continue

            stats['total_frames_analyzed'] += 1
            n_animals = len(occ_result['track_ids'])

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

            if occ_result['num_occluded_pairs'] > 0:
                stats['frames_with_occlusion'] += 1

            for j, track_id in enumerate(track_ids):
                occlusion_received = np.sum(occ_matrix[:, j])
                if occlusion_received > 0:
                    stats['per_track_exposure'][track_id].append(occlusion_received)
                    if occlusion_received > OCCLUSION_THRESHOLD:
                        all_occlusion_values.append(occlusion_received)

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

        if stats['frames_with_multiple_animals'] > 0:
            stats['occlusion_rate'] = (
                stats['frames_with_occlusion'] / stats['frames_with_multiple_animals']
            )

        if all_occlusion_values:
            stats['mean_severity'] = float(np.mean(all_occlusion_values))
            stats['max_severity'] = float(np.max(all_occlusion_values))

        stats['per_track_exposure'] = {
            track_id: float(np.mean(values)) if values else 0.0
            for track_id, values in stats['per_track_exposure'].items()
        }

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

    # ========================================================================
    # TRACKLET VIEWPOINT CHARACTERIZATION (UPDATED FOR V4)
    # ========================================================================

    def compute_tracklet_viewpoint_profiles(self, occlusion_stats=None):
        """
        Comprehensive per-tracklet viewpoint characterization.

        V4 Changes:
            - Uses NMS for diverse exemplar selection (up to CROPS_PER_FACE per orientation)
            - Removed reid_readiness_score and completeness_score
            - All orientations treated equally
        """
        print("\n" + "="*70)
        print("COMPUTING TRACKLET VIEWPOINT PROFILES (V4)")
        print("="*70)

        semantic_labels = CONFIG['ALL_FACES']
        visible_labels = CONFIG['VISIBLE_FACES']
        tracklet_profiles = {}

        # Build per-track, per-frame occlusion lookup
        occlusion_lookup = {}
        if occlusion_stats and 'temporal_profile' in occlusion_stats:
            for frame_data in occlusion_stats['temporal_profile']:
                frame_name = frame_data.get('frame')
                if not frame_name:
                    continue
                for pair in frame_data.get('occluder_occludee_pairs', []):
                    occludee = pair.get('occludee')
                    severity = pair.get('severity', 0.0)
                    if occludee is not None:
                        if occludee not in occlusion_lookup:
                            occlusion_lookup[occludee] = {}
                        if frame_name not in occlusion_lookup[occludee]:
                            occlusion_lookup[occludee][frame_name] = severity
                        else:
                            occlusion_lookup[occludee][frame_name] = max(
                                occlusion_lookup[occludee][frame_name], severity
                            )
            if occlusion_lookup:
                print(f"  Loaded occlusion data for {len(occlusion_lookup)} tracks")

        for track_id in self.labeled_tracks:
            if track_id not in self.semantic_faces:
                continue

            print(f"\nAnalyzing Track {track_id}...")

            visibility_matrix = {label: [] for label in semantic_labels}
            quality_matrix = {label: [] for label in semantic_labels}
            frame_quality_scores = {}

            # Get sorted frame names with indices
            sorted_frames = sorted(
                self.semantic_faces[track_id].keys(),
                key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0
            )

            frame_name_to_idx = {name: idx for idx, name in enumerate(sorted_frames)}

            for frame_name in sorted_frames:
                camera_params = self._load_camera_params(frame_name)
                if not camera_params:
                    continue

                img_shape = (480, 640, 3)
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
                        visibility_matrix[label].append(0)
                        quality_matrix[label].append(0.0)
                        frame_qualities[label] = 0.0
                    elif label in semantic_faces:
                        is_visible, visibility_score = self._calculate_face_visibility(
                            semantic_faces[label], camera_params
                        )
                        visibility_matrix[label].append(1 if is_visible else 0)

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

            total_frames = len(self.semantic_faces[track_id])

            # Compute coverage vector
            coverage_vector = {
                label: sum(visibility_matrix[label]) / total_frames if total_frames > 0 else 0.0
                for label in semantic_labels
            }
            coverage_vector['bottom'] = float('nan')

            # Compute diversity index (V4: equal treatment of all visible faces)
            coverage_values = [coverage_vector[label] for label in visible_labels]
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

            # V4: Select diverse exemplars using NMS
            exemplars = {}

            for label in semantic_labels:
                if label == 'bottom':
                    exemplars[label] = []
                    continue

                # Build quality data for NMS: (frame_idx, frame_name, quality_score)
                quality_data = []
                for frame_idx, frame_name in enumerate(sorted_frames):
                    quality = quality_matrix[label][frame_idx] if frame_idx < len(quality_matrix[label]) else 0.0

                    # Apply occlusion penalty
                    track_id_int = int(track_id) if isinstance(track_id, str) else track_id
                    occlusion_severity = 0.0
                    if track_id_int in occlusion_lookup:
                        occlusion_severity = occlusion_lookup[track_id_int].get(frame_name, 0.0)

                    # Penalize quality by occlusion
                    occlusion_penalty = min(occlusion_severity * 2.0, 0.95)
                    effective_quality = quality * (1.0 - occlusion_penalty)

                    if effective_quality > 0:
                        quality_data.append((frame_idx, frame_name, effective_quality))

                # Apply NMS to select diverse exemplars
                exemplars[label] = self._select_diverse_exemplars(quality_data)

                if exemplars[label]:
                    print(f"    [{label}] Selected {len(exemplars[label])} diverse exemplars")

            # Identify coverage gaps
            coverage_threshold = CONFIG['THRESHOLDS']['coverage_fair']
            coverage_gaps = [label for label in visible_labels
                           if coverage_vector[label] < coverage_threshold]

            # Store profile (V4 schema)
            tracklet_profiles[track_id] = {
                'track_id': track_id,
                'total_frames': total_frames,
                'coverage_vector': coverage_vector,
                'diversity_index': diversity_index,
                'normalized_diversity': normalized_diversity,
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
                'exemplars': {
                    label: [{'frame': e.frame, 'quality_score': e.quality_score, 'frame_idx': e.frame_idx}
                           for e in exemplars[label]]
                    for label in semantic_labels
                },
                'coverage_gaps': coverage_gaps,
                'frame_quality_scores': frame_quality_scores,
            }

            visible_coverage_str = [f'{coverage_vector[l]:.2f}' for l in visible_labels]
            print(f"  Coverage Vector (visible): {visible_coverage_str}")
            print(f"  (Bottom: N/A - drone footage)")
            print(f"  Diversity Index: {normalized_diversity:.3f}")
            if coverage_gaps:
                print(f"  Coverage Gaps: {coverage_gaps}")

        return tracklet_profiles

    def print_analysis_summary(self, tracklet_profiles):
        """Print concise analysis summary."""
        if not tracklet_profiles:
            print("No tracklet profiles available.")
            return

        print("\n" + "="*70)
        print("VIEWPOINT ANALYSIS SUMMARY (V4)")
        print("="*70)

        visible_labels = CONFIG['VISIBLE_FACES']

        print(f"\nANALYZED TRACKLETS: {len(tracklet_profiles)}")

        for track_id in sorted(tracklet_profiles.keys()):
            profile = tracklet_profiles[track_id]
            print(f"\n  TRACK {track_id}:")
            print(f"    Frames: {profile['total_frames']}")
            coverage_values = [f'{profile["coverage_vector"][l]:.2f}' for l in visible_labels]
            print(f"    Coverage: {coverage_values}")
            print(f"    Diversity Index: {profile['normalized_diversity']:.3f}")

            # Count exemplars per face
            exemplar_counts = {l: len(profile['exemplars'].get(l, [])) for l in visible_labels}
            print(f"    Exemplars: {exemplar_counts}")

            if profile['coverage_gaps']:
                print(f"    Coverage Gaps: {', '.join(profile['coverage_gaps'])}")

        avg_diversity = np.mean([p['normalized_diversity'] for p in tracklet_profiles.values()])

        print(f"\nDATASET AVERAGES:")
        print(f"  Diversity Index: {avg_diversity:.3f}")

        print("\n" + "="*70)


# =============================================================================
# QUALITY GRADER (SIMPLIFIED FOR V4 - GEOMETRIC ONLY)
# =============================================================================

class QualityGrader:
    """
    Converts raw metrics to geometric grades (V4: no subjective scores).

    Grade thresholds:
        A (Excellent): >= 0.8 or 5/5 views
        B (Good):      >= 0.6 or 4/5 views
        C (Fair):      >= 0.4 or 3/5 views
        F (Poor):      < 0.4 or <3 views
    """

    GRADE_COLORS = {
        'A': '#2ECC71',  # Green
        'B': '#3498DB',  # Blue
        'C': '#F39C12',  # Orange
        'F': '#E74C3C',  # Red
    }

    GRADE_LABELS = {
        'A': 'Excellent',
        'B': 'Good',
        'C': 'Fair',
        'F': 'Poor',
    }

    def __init__(self):
        self.thresholds = CONFIG['THRESHOLDS']

    def _score_to_grade(self, score: float, thresholds: Dict = None) -> str:
        """Convert a 0-1 score to a letter grade."""
        if thresholds is None:
            thresholds = {'A': 0.8, 'B': 0.6, 'C': 0.4}

        if score >= thresholds.get('A', 0.8):
            return 'A'
        elif score >= thresholds.get('B', 0.6):
            return 'B'
        elif score >= thresholds.get('C', 0.4):
            return 'C'
        else:
            return 'F'

    def grade_coverage(self, coverage_vector: Dict[str, float]) -> GradeResult:
        """Grade viewpoint coverage (geometric only)."""
        visible_labels = CONFIG['VISIBLE_FACES']

        # Count views with sufficient coverage
        good_threshold = self.thresholds.get('coverage_good', 0.15)
        good_views = sum(1 for label in visible_labels
                        if coverage_vector.get(label, 0) >= good_threshold)

        # Grade based on number of good views (V4: equal weight for all)
        if good_views >= 5:
            letter = 'A'
            description = f"All {good_views}/5 orientations have good coverage"
        elif good_views >= 4:
            letter = 'B'
            description = f"{good_views}/5 orientations covered"
        elif good_views >= 3:
            letter = 'C'
            description = f"Only {good_views}/5 orientations covered"
        else:
            letter = 'F'
            description = f"Poor coverage: only {good_views}/5 orientations"

        gaps = [label for label in visible_labels
                if coverage_vector.get(label, 0) < self.thresholds.get('coverage_fair', 0.08)]
        if gaps:
            description += f" (gaps: {', '.join(gaps)})"

        return GradeResult(
            letter=letter,
            score=good_views / 5.0,
            label=self.GRADE_LABELS[letter],
            description=description,
            color=self.GRADE_COLORS[letter]
        )

    def grade_diversity(self, normalized_diversity: float) -> GradeResult:
        """Grade viewpoint diversity (geometric only)."""
        thresholds = {
            'A': self.thresholds.get('diversity_good', 0.60) + 0.2,
            'B': self.thresholds.get('diversity_good', 0.60),
            'C': self.thresholds.get('diversity_fair', 0.40)
        }
        letter = self._score_to_grade(normalized_diversity, thresholds)

        descriptions = {
            'A': "Views evenly distributed across orientations",
            'B': "Good viewpoint spread with minor imbalance",
            'C': "Viewpoints concentrated in few orientations",
            'F': "Very uneven viewpoint distribution",
        }

        return GradeResult(
            letter=letter,
            score=normalized_diversity,
            label=self.GRADE_LABELS[letter],
            description=descriptions[letter],
            color=self.GRADE_COLORS[letter]
        )

    def grade_quality(self, avg_quality: float,
                      quality_per_orientation: Dict[str, float] = None) -> GradeResult:
        """Grade overall image quality (geometric only)."""
        letter = self._score_to_grade(avg_quality)

        descriptions = {
            'A': "High quality views with good visibility",
            'B': "Acceptable quality for most applications",
            'C': "Quality issues may affect analysis",
            'F': "Poor quality - distant or blurry captures",
        }

        description = descriptions[letter]

        if quality_per_orientation:
            low_quality = [label for label, q in quality_per_orientation.items()
                          if not np.isnan(q) and 0 < q < 0.3 and label != 'bottom']
            if low_quality:
                description += f" (low quality: {', '.join(low_quality)})"

        return GradeResult(
            letter=letter,
            score=avg_quality,
            label=self.GRADE_LABELS[letter],
            description=description,
            color=self.GRADE_COLORS[letter]
        )

    def compute_overall_grade(self, grades: TrackletGrades) -> GradeResult:
        """Compute overall grade (V4: equal weights for geometric metrics)."""
        # V4: Equal weights for all geometric metrics
        weighted_score = (
            grades.coverage.score / 3.0 +
            grades.diversity.score / 3.0 +
            grades.quality.score / 3.0
        )

        letter = self._score_to_grade(weighted_score)

        descriptions = {
            'A': "Excellent geometric coverage",
            'B': "Good geometric coverage",
            'C': "Fair geometric coverage with limitations",
            'F': "Poor geometric coverage",
        }

        return GradeResult(
            letter=letter,
            score=weighted_score,
            label=self.GRADE_LABELS[letter],
            description=descriptions[letter],
            color=self.GRADE_COLORS[letter]
        )

    def grade_tracklet(self, profile: Dict) -> TrackletGrades:
        """Grade all geometric aspects of a tracklet (V4: no subjective scores)."""
        coverage_vector = profile.get('coverage_vector', {})
        normalized_diversity = profile.get('normalized_diversity', 0.0)
        quality_per_orientation = profile.get('average_quality_per_orientation', {})

        valid_qualities = [q for label, q in quality_per_orientation.items()
                         if label != 'bottom' and not np.isnan(q) and q > 0]
        avg_quality = np.mean(valid_qualities) if valid_qualities else 0.0

        coverage_grade = self.grade_coverage(coverage_vector)
        diversity_grade = self.grade_diversity(normalized_diversity)
        quality_grade = self.grade_quality(avg_quality, quality_per_orientation)

        grades = TrackletGrades(
            overall=None,
            coverage=coverage_grade,
            diversity=diversity_grade,
            quality=quality_grade,
        )

        grades.overall = self.compute_overall_grade(grades)

        return grades


# =============================================================================
# MASK CROP EXTRACTOR (UPDATED FOR V4 - SQUARE PADDING)
# =============================================================================

class MaskCropExtractor:
    """Extracts masked animal crops from images (V4: square padding, error resilience)."""

    def __init__(self, images_dir: Path, mask_dir: Path, results_dir: Path):
        self.images_dir = Path(images_dir) if images_dir else None
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.results_dir = Path(results_dir) if results_dir else None

        self.mask_track_mapping = self._load_mask_track_mapping()

    def _load_mask_track_mapping(self) -> Dict:
        """Load mask-to-track mapping from results directory."""
        if self.results_dir is None:
            return {}

        possible_paths = [
            self.results_dir / "mask_track_mapping.json",
            self.results_dir / "corrected_labels" / "mask_track_mapping.json",
            self.results_dir.parent / "mask_track_mapping.json",
        ]

        for mapping_file in possible_paths:
            if mapping_file.exists():
                with open(mapping_file, 'r') as f:
                    print(f"  Loaded mask-track mapping from: {mapping_file}")
                    return json.load(f)

        return {}

    def _load_image(self, frame_name: str) -> Optional[np.ndarray]:
        """Load image for a frame."""
        if self.images_dir is None:
            return None

        frame_num = re.findall(r'\d+', str(frame_name))
        frame_key = frame_num[0] if frame_num else frame_name

        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            for name in [frame_name, frame_key]:
                img_path = self.images_dir / f"{name}{ext}"
                if img_path.exists():
                    return cv2.imread(str(img_path))

        return None

    def _load_mask(self, frame_name: str, track_id: int) -> Optional[np.ndarray]:
        """Load mask for a specific track in a frame."""
        if self.mask_dir is None or not PYCOCOTOOLS_AVAILABLE:
            return None

        frame_num = re.findall(r'\d+', str(frame_name))
        frame_key = frame_num[0] if frame_num else frame_name

        mask_idx = self.mask_track_mapping.get(str(frame_key), {}).get(str(track_id))

        json_file = None
        for name in [frame_key, frame_name]:
            candidate = self.mask_dir / f"{name}_results.json"
            if candidate.exists():
                json_file = candidate
                break

        if json_file is None:
            return None

        try:
            with open(json_file, 'r') as f:
                results = json.load(f)

            annotations = results.get('annotations', [])

            if mask_idx is not None and mask_idx < len(annotations):
                ann = annotations[mask_idx]
                rle = ann.get('segmentation')
                if rle:
                    return mask_utils.decode(rle)

            if len(annotations) == 1:
                rle = annotations[0].get('segmentation')
                if rle:
                    return mask_utils.decode(rle)

        except Exception as e:
            print(f"Warning: Could not load mask for {frame_name}, track {track_id}: {e}")

        return None

    def _create_error_image(self, size: int = 200) -> np.ndarray:
        """Create an error placeholder image (V4: error resilience)."""
        img = np.zeros((size, size, 3), dtype=np.uint8)
        # Red background
        img[:, :, 2] = 80  # Dark red

        # Add error text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "IMG ERR"
        text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
        text_x = (size - text_size[0]) // 2
        text_y = (size + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), font, 0.6, (255, 255, 255), 2)

        return img

    def extract_crop(self, frame_name: str, track_id: int,
                    bbox_data: Dict = None, padding: int = None,
                    background: str = None) -> Optional[np.ndarray]:
        """
        Extract a masked crop of an animal (V4: square padding, error resilience).

        Args:
            frame_name: Frame identifier
            track_id: Track ID
            bbox_data: Optional bounding box data (for crop bounds)
            padding: Pixels to add around the crop (default from CONFIG)
            background: 'white' or 'transparent' (default from CONFIG)

        Returns:
            Cropped image with mask applied, or error placeholder if unavailable
        """
        if padding is None:
            padding = CONFIG['CROP_PADDING']
        if background is None:
            background = CONFIG['CROP_BG_COLOR']

        try:
            # Load image
            image = self._load_image(frame_name)
            if image is None:
                return self._create_error_image()

            # Load mask
            mask = self._load_mask(frame_name, track_id)
            if mask is None:
                if bbox_data is not None:
                    return self._crop_without_mask(image, bbox_data, padding)
                return self._create_error_image()

            # Find bounding box of mask
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                return self._create_error_image()

            # V4: Square padding for uniform crops
            mask_x1, mask_x2 = xs.min(), xs.max()
            mask_y1, mask_y2 = ys.min(), ys.max()

            mask_w = mask_x2 - mask_x1
            mask_h = mask_y2 - mask_y1
            mask_cx = (mask_x1 + mask_x2) // 2
            mask_cy = (mask_y1 + mask_y2) // 2

            # Create square crop centered on mask
            max_dim = max(mask_w, mask_h) + 2 * padding
            half_dim = max_dim // 2

            # Calculate crop bounds (clip to image dimensions)
            img_h, img_w = image.shape[:2]
            x1 = max(0, mask_cx - half_dim)
            x2 = min(img_w, mask_cx + half_dim)
            y1 = max(0, mask_cy - half_dim)
            y2 = min(img_h, mask_cy + half_dim)

            # Crop image and mask
            cropped_image = image[y1:y2, x1:x2].copy()
            cropped_mask = mask[y1:y2, x1:x2]

            # Apply mask
            if background == 'white':
                result = np.ones_like(cropped_image) * 255
                result[cropped_mask > 0] = cropped_image[cropped_mask > 0]
            else:
                result = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2BGRA)
                result[:, :, 3] = (cropped_mask * 255).astype(np.uint8)

            return result

        except Exception as e:
            print(f"Warning: Crop extraction failed for {frame_name}, track {track_id}: {e}")
            return self._create_error_image()

    def _crop_without_mask(self, image: np.ndarray, bbox_data: Dict,
                          padding: int) -> np.ndarray:
        """Crop image using bounding box when mask is unavailable."""
        try:
            center = bbox_data.get('center', None)
            if center is None:
                return self._create_error_image()

            h, w = image.shape[:2]
            cx, cy = int(w / 2), int(h / 2)

            crop_size = 200
            x1 = max(0, cx - crop_size - padding)
            x2 = min(w, cx + crop_size + padding)
            y1 = max(0, cy - crop_size - padding)
            y2 = min(h, cy + crop_size + padding)

            return image[y1:y2, x1:x2].copy()
        except Exception:
            return self._create_error_image()


# =============================================================================
# PDF REPORT GENERATOR (V4: FILMSTRIP LAYOUT)
# =============================================================================

class PDFReportGenerator:
    """Generates Portrait/Filmstrip PDF reports (V4 layout)."""

    def __init__(self, analyzer: ViewpointAnalyzer,
                 grader: QualityGrader = None,
                 crop_extractor: MaskCropExtractor = None):
        self.analyzer = analyzer
        self.grader = grader or QualityGrader()
        self.crop_extractor = crop_extractor

    def generate_report(self, tracklet_profiles: Dict,
                       occlusion_stats: Dict = None,
                       output_path: Path = None) -> Path:
        """
        Generate Filmstrip PDF report (V4 layout).

        Args:
            tracklet_profiles: Output from ViewpointAnalyzer.compute_tracklet_viewpoint_profiles()
            occlusion_stats: Optional occlusion statistics
            output_path: Output PDF path

        Returns:
            Path to generated PDF
        """
        if output_path is None:
            output_path = self.analyzer.annotator_output_dir / "viewpoint_analysis" / "tracklet_filmstrip_report.pdf"

        output_path.parent.mkdir(exist_ok=True)

        print("\n" + "="*70)
        print("GENERATING FILMSTRIP PDF REPORT (V4)")
        print("="*70)

        # Grade all tracklets
        all_grades = {}
        for track_id, profile in tracklet_profiles.items():
            all_grades[track_id] = self.grader.grade_tracklet(profile)

        # Generate PDF
        with PdfPages(str(output_path)) as pdf:
            # Page 1: Executive Summary
            self._create_summary_page(pdf, tracklet_profiles, all_grades)

            # Per-tracklet filmstrip pages
            for track_id in sorted(tracklet_profiles.keys()):
                self._create_filmstrip_page(
                    pdf, track_id,
                    tracklet_profiles[track_id],
                    all_grades[track_id]
                )

            # Occlusion page (if available)
            if occlusion_stats and occlusion_stats.get('frames_with_multiple_animals', 0) > 0:
                self._create_occlusion_page(pdf, occlusion_stats)

        print(f"\nSaved PDF report to: {output_path}")
        return output_path

    def _create_summary_page(self, pdf: PdfPages, tracklet_profiles: Dict,
                            all_grades: Dict):
        """Create executive summary page."""
        fig = plt.figure(figsize=CONFIG['PDF_FIGURE_SIZE'])

        # Title
        fig.suptitle('TRACKLET VIEWPOINT REPORT (V4)', fontsize=18, fontweight='bold', y=0.98)

        # Subtitle
        scene_name = self.analyzer.annotator_output_dir.name
        date_str = datetime.now().strftime('%Y-%m-%d')
        fig.text(0.5, 0.94, f'Scene: {scene_name}  |  Date: {date_str}',
                ha='center', fontsize=10, style='italic')

        gs = GridSpec(3, 2, figure=fig, height_ratios=[0.25, 0.5, 0.25],
                     left=0.08, right=0.92, top=0.90, bottom=0.08,
                     hspace=0.3, wspace=0.3)

        # Summary stats
        ax_stats = fig.add_subplot(gs[0, :])
        ax_stats.axis('off')

        total_tracks = len(tracklet_profiles)
        total_frames = sum(p.get('total_frames', 0) for p in tracklet_profiles.values())

        avg_score = np.mean([g.overall.score for g in all_grades.values()])
        if avg_score >= 0.75:
            dataset_grade = 'A'
        elif avg_score >= 0.55:
            dataset_grade = 'B'
        elif avg_score >= 0.35:
            dataset_grade = 'C'
        else:
            dataset_grade = 'F'

        grade_color = self.grader.GRADE_COLORS[dataset_grade]

        stats_text = (
            f"Tracks: {total_tracks}    "
            f"Frames: {total_frames}    "
            f"Overall Grade: {dataset_grade}"
        )
        ax_stats.text(0.5, 0.5, stats_text, ha='center', va='center',
                     fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor=grade_color, alpha=0.3))

        # Tracklet table
        ax_table = fig.add_subplot(gs[1, :])
        ax_table.axis('off')

        visible_labels = CONFIG['VISIBLE_FACES']

        table_data = [['Track', 'Grade', 'Diversity'] + [l.capitalize()[:3] for l in visible_labels]]
        cell_colors = [['#E8E8E8'] * (3 + len(visible_labels))]

        for track_id in sorted(tracklet_profiles.keys()):
            profile = tracklet_profiles[track_id]
            grades = all_grades[track_id]

            coverage = profile.get('coverage_vector', {})
            exemplars = profile.get('exemplars', {})

            # Count exemplars per face
            exemplar_counts = [str(len(exemplars.get(l, []))) for l in visible_labels]

            row = [
                f"Track {track_id}",
                grades.overall.letter,
                f"{grades.diversity.score:.2f}",
            ] + exemplar_counts

            table_data.append(row)

            row_color = self.grader.GRADE_COLORS[grades.overall.letter]
            cell_colors.append([row_color + '40'] * (3 + len(visible_labels)))

        table = ax_table.table(
            cellText=table_data,
            cellLoc='center',
            loc='center',
            cellColours=cell_colors
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)

        # Configuration info
        ax_config = fig.add_subplot(gs[2, :])
        ax_config.axis('off')

        config_text = (
            f"V4 Configuration:\n"
            f"  Crops per face: {CONFIG['CROPS_PER_FACE']}\n"
            f"  NMS frame gap: {CONFIG['NMS_FRAME_GAP']}\n"
            f"  Coverage threshold (good): {CONFIG['THRESHOLDS']['coverage_good']:.0%}"
        )
        ax_config.text(0.05, 0.9, config_text, va='top', fontsize=9,
                      fontfamily='monospace',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_filmstrip_page(self, pdf: PdfPages, track_id: int,
                              profile: Dict, grades: TrackletGrades):
        """Create filmstrip page for a single tracklet (V4 layout)."""
        fig = plt.figure(figsize=CONFIG['PDF_FIGURE_SIZE'])

        visible_labels = CONFIG['VISIBLE_FACES']
        n_crops = CONFIG['CROPS_PER_FACE']

        # Header area (top 15%)
        # Main content: 5x5 grid (85%)
        gs = GridSpec(6, n_crops + 1, figure=fig,
                     height_ratios=[0.15] + [0.17] * 5,
                     width_ratios=[0.12] + [0.176] * n_crops,
                     left=0.02, right=0.98, top=0.95, bottom=0.02,
                     hspace=0.05, wspace=0.05)

        # Header
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.axis('off')

        header_text = (
            f"TRACK {track_id}  |  "
            f"Grade: {grades.overall.letter} ({grades.overall.label})  |  "
            f"Frames: {profile['total_frames']}  |  "
            f"Diversity: {profile['normalized_diversity']:.2f}"
        )
        ax_header.text(0.5, 0.5, header_text, ha='center', va='center',
                      fontsize=12, fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.3',
                               facecolor=grades.overall.color, alpha=0.3))

        # Filmstrip grid
        exemplars = profile.get('exemplars', {})

        for row_idx, label in enumerate(visible_labels):
            # Row label
            ax_label = fig.add_subplot(gs[row_idx + 1, 0])
            ax_label.axis('off')
            ax_label.text(0.5, 0.5, label.upper(), ha='center', va='center',
                         fontsize=10, fontweight='bold', rotation=90)

            # Get exemplars for this orientation
            label_exemplars = exemplars.get(label, [])

            for col_idx in range(n_crops):
                ax_crop = fig.add_subplot(gs[row_idx + 1, col_idx + 1])

                if col_idx < len(label_exemplars):
                    exemplar = label_exemplars[col_idx]
                    frame_name = exemplar.get('frame', exemplar) if isinstance(exemplar, dict) else exemplar
                    quality = exemplar.get('quality_score', 0) if isinstance(exemplar, dict) else 0

                    # Extract crop
                    crop_image = None
                    if self.crop_extractor:
                        crop_image = self.crop_extractor.extract_crop(frame_name, track_id)

                    if crop_image is not None:
                        if crop_image.shape[-1] == 3:
                            crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
                        elif crop_image.shape[-1] == 4:
                            crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGRA2RGBA)
                        ax_crop.imshow(crop_image)

                        # Frame annotation
                        frame_num = re.findall(r'\d+', str(frame_name))
                        frame_str = frame_num[0] if frame_num else frame_name
                        ax_crop.text(0.02, 0.02, f'F:{frame_str}\nQ:{quality:.2f}',
                                    transform=ax_crop.transAxes, fontsize=6,
                                    verticalalignment='bottom',
                                    bbox=dict(boxstyle='round,pad=0.1',
                                             facecolor='white', alpha=0.7))
                    else:
                        # No crop available
                        ax_crop.set_facecolor('#E0E0E0')
                        ax_crop.text(0.5, 0.5, 'N/A', ha='center', va='center',
                                   fontsize=10, color='gray')
                else:
                    # Placeholder for missing exemplar
                    ax_crop.set_facecolor('#F5F5F5')
                    ax_crop.text(0.5, 0.5, 'N/A', ha='center', va='center',
                               fontsize=10, color='lightgray')

                ax_crop.axis('off')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_occlusion_page(self, pdf: PdfPages, occlusion_stats: Dict):
        """Create occlusion analysis page."""
        fig = plt.figure(figsize=CONFIG['PDF_FIGURE_SIZE'])
        fig.suptitle('INTER-ANIMAL OCCLUSION', fontsize=16, fontweight='bold', y=0.98)

        gs = GridSpec(2, 2, figure=fig,
                     left=0.1, right=0.9, top=0.90, bottom=0.1,
                     hspace=0.3, wspace=0.3)

        occlusion_rate = occlusion_stats.get('occlusion_rate', 0)

        # Determine grade
        severe_threshold = CONFIG['THRESHOLDS']['occlusion_severe']
        if occlusion_rate <= 0.10:
            grade_letter = 'A'
        elif occlusion_rate <= 0.20:
            grade_letter = 'B'
        elif occlusion_rate <= severe_threshold:
            grade_letter = 'C'
        else:
            grade_letter = 'F'

        # Summary
        ax_summary = fig.add_subplot(gs[0, 0])
        ax_summary.axis('off')

        summary_text = (
            f"Occlusion Rate: {occlusion_rate:.1%}\n"
            f"Grade: {grade_letter}\n\n"
            f"Frames with 2+ Animals: {occlusion_stats.get('frames_with_multiple_animals', 0)}\n"
            f"Frames with Occlusion: {occlusion_stats.get('frames_with_occlusion', 0)}\n\n"
            f"Mean Severity: {occlusion_stats.get('mean_severity', 0):.3f}\n"
            f"Max Severity: {occlusion_stats.get('max_severity', 0):.3f}"
        )
        ax_summary.text(0.05, 0.95, summary_text, va='top', fontsize=10,
                       fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.5',
                                facecolor=self.grader.GRADE_COLORS[grade_letter], alpha=0.3))

        # Per-track exposure
        ax_exposure = fig.add_subplot(gs[0, 1])
        per_track = occlusion_stats.get('per_track_exposure', {})

        if per_track:
            track_ids = sorted(per_track.keys())
            exposures = [per_track[tid] for tid in track_ids]

            colors = ['#E74C3C' if e > severe_threshold else '#F39C12' if e > 0.1 else '#2ECC71'
                     for e in exposures]
            ax_exposure.bar(range(len(track_ids)), exposures, color=colors)
            ax_exposure.set_xticks(range(len(track_ids)))
            ax_exposure.set_xticklabels([f'T{tid}' for tid in track_ids])
            ax_exposure.set_ylabel('Mean Occlusion')
            ax_exposure.set_title('Per-Track Exposure')
        else:
            ax_exposure.text(0.5, 0.5, 'No multi-animal frames', ha='center', va='center')
            ax_exposure.set_title('Per-Track Exposure')

        # Temporal profile
        ax_temporal = fig.add_subplot(gs[1, :])
        temporal = occlusion_stats.get('temporal_profile', [])

        if temporal:
            multi_frames = [t for t in temporal if t.get('num_animals', 0) >= 2]
            if multi_frames:
                max_occs = [t.get('max_occlusion', 0) for t in multi_frames]
                ax_temporal.fill_between(range(len(max_occs)), max_occs, alpha=0.3, color='red')
                ax_temporal.plot(max_occs, color='red', linewidth=1)
                ax_temporal.set_xlabel('Multi-Animal Frame Index')
                ax_temporal.set_ylabel('Max Occlusion')
                ax_temporal.set_ylim(0, 1)

        ax_temporal.set_title('Occlusion Over Time')
        ax_temporal.grid(True, alpha=0.3)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def save_json_report(self, tracklet_profiles: Dict,
                        all_grades: Dict,
                        output_path: Path = None) -> Path:
        """Save JSON report with V4 schema."""
        if output_path is None:
            output_path = self.analyzer.annotator_output_dir / "viewpoint_analysis" / "tracklet_report_v4.json"

        output_path.parent.mkdir(exist_ok=True)

        report_data = {}

        for track_id, profile in tracklet_profiles.items():
            grades = all_grades.get(track_id)

            report_data[str(track_id)] = {
                'track_id': track_id,
                'metrics': {
                    'total_frames': profile['total_frames'],
                    'coverage_vector': self._convert_numpy_types(profile['coverage_vector']),
                    'diversity_index': float(profile['normalized_diversity']),
                },
                'grades': {
                    'overall': {'letter': grades.overall.letter, 'score': float(grades.overall.score)},
                    'coverage': {'letter': grades.coverage.letter, 'score': float(grades.coverage.score)},
                    'diversity': {'letter': grades.diversity.letter, 'score': float(grades.diversity.score)},
                    'quality': {'letter': grades.quality.letter, 'score': float(grades.quality.score)},
                } if grades else {},
                'exemplars': self._convert_numpy_types(profile['exemplars']),
                'coverage_gaps': profile['coverage_gaps'],
            }

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"Saved JSON report to: {output_path}")
        return output_path

    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        else:
            return obj


# =============================================================================
# MAIN ENTRY POINT (V4)
# =============================================================================

def main():
    """Command-line interface for Viewpoint Analyzer v4."""
    parser = argparse.ArgumentParser(
        description="Viewpoint Analyzer v4 - Objective Filmstrip Reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (generates filmstrip PDF)
    python viewpoint_analyzer_v4.py --annotator_output results/zebra/scene1/corrected/

    # With mask crops
    python viewpoint_analyzer_v4.py \\
        --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/ \\
        --mask_dir data/zebra/scene1/grounded-sam/

    # Include occlusion analysis
    python viewpoint_analyzer_v4.py \\
        --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/ \\
        --compute_occlusion
        """
    )

    parser.add_argument(
        "--annotator_output",
        required=True,
        help="Path to annotator tool output directory"
    )

    parser.add_argument(
        "--images_dir",
        default=None,
        help="Path to original images"
    )

    parser.add_argument(
        "--mask_dir",
        default=None,
        help="Path to grounded-SAM mask directory"
    )

    parser.add_argument(
        "--results_dir",
        default=None,
        help="Path to results directory"
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for results"
    )

    parser.add_argument(
        "--compute_occlusion",
        action="store_true",
        help="Compute inter-animal occlusion statistics"
    )

    parser.add_argument(
        "--skip_pdf",
        action="store_true",
        help="Skip generating PDF report"
    )

    parser.add_argument(
        "--crops_per_face",
        type=int,
        default=None,
        help=f"Number of exemplar crops per face (default: {CONFIG['CROPS_PER_FACE']})"
    )

    parser.add_argument(
        "--nms_gap",
        type=int,
        default=None,
        help=f"Minimum frame gap for NMS (default: {CONFIG['NMS_FRAME_GAP']})"
    )

    args = parser.parse_args()

    # Update CONFIG from args
    if args.crops_per_face:
        CONFIG['CROPS_PER_FACE'] = args.crops_per_face
    if args.nms_gap:
        CONFIG['NMS_FRAME_GAP'] = args.nms_gap

    print("=" * 70)
    print("VIEWPOINT ANALYZER v4")
    print("Objective Filmstrip Reporting")
    print("=" * 70)
    print(f"  Crops per face: {CONFIG['CROPS_PER_FACE']}")
    print(f"  NMS frame gap: {CONFIG['NMS_FRAME_GAP']}")

    try:
        analyzer = ViewpointAnalyzer(
            annotator_output_dir=args.annotator_output,
            images_dir=args.images_dir
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return 1

    # Occlusion analysis
    occlusion_stats = None
    if args.compute_occlusion:
        occlusion_stats = analyzer.compute_occlusion_statistics()

    # Compute viewpoint profiles
    tracklet_profiles = analyzer.compute_tracklet_viewpoint_profiles(occlusion_stats)

    # Print summary
    analyzer.print_analysis_summary(tracklet_profiles)

    # Set up mask crop extractor
    crop_extractor = None
    if args.images_dir:
        mask_dir = args.mask_dir
        if mask_dir is None:
            possible_dirs = [
                Path(args.images_dir) / "grounded-sam",  # grounded-sam inside images dir
                Path(args.images_dir).parent / "grounded-sam",
                Path(args.annotator_output).parent.parent / "grounded-sam",
                Path(args.annotator_output).parent / "grounded-sam",
            ]
            for candidate in possible_dirs:
                if candidate.exists():
                    mask_dir = str(candidate)
                    print(f"  Auto-detected mask directory: {mask_dir}")
                    break

        results_dir = args.results_dir
        if results_dir is None:
            for candidate in [Path(args.annotator_output).parent, Path(args.annotator_output)]:
                if (candidate / "mask_track_mapping.json").exists():
                    results_dir = str(candidate)
                    break

        if mask_dir and PYCOCOTOOLS_AVAILABLE:
            crop_extractor = MaskCropExtractor(
                images_dir=Path(args.images_dir),
                mask_dir=Path(mask_dir),
                results_dir=Path(results_dir) if results_dir else Path(args.annotator_output).parent
            )
            print(f"  Mask crop extraction enabled")

    # Generate reports
    if not args.skip_pdf and tracklet_profiles:
        grader = QualityGrader()

        all_grades = {}
        for track_id, profile in tracklet_profiles.items():
            all_grades[track_id] = grader.grade_tracklet(profile)

        report_generator = PDFReportGenerator(
            analyzer=analyzer,
            grader=grader,
            crop_extractor=crop_extractor
        )

        # Generate PDF
        report_generator.generate_report(
            tracklet_profiles=tracklet_profiles,
            occlusion_stats=occlusion_stats
        )

        # Save JSON
        report_generator.save_json_report(
            tracklet_profiles=tracklet_profiles,
            all_grades=all_grades
        )

    output_dir = args.output_dir or (args.annotator_output + "/viewpoint_analysis")

    print("\n" + "=" * 70)
    print(f"Analysis complete. Results saved to: {output_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
