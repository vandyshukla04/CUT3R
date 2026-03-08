#!/usr/bin/env python3
"""
Viewpoint Analyzer v6 for WildLIFT Pipeline

Publication-quality aggregate PDF with all tracklets combined.

Key Changes from v5:
    - Aggregate PDF: Combines all tracklets into a single publication-quality figure
    - Coverage matrix: First page shows overview of all tracks/orientations
    - Compact filmstrip: Horizontal strips per track, no wasted space for missing views
    - Modern/Colorful style: Color-coded orientations with gradient accents
    - Load from saved: Regenerate PDFs from existing selection files

Usage:
    # Generate aggregate PDF from saved selections
    python viewpoint_analyzer_v6.py --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/ --load_saved --aggregate

    # Interactive selection + aggregate PDF
    python viewpoint_analyzer_v6.py --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/ --aggregate
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
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from collections import defaultdict, Counter
from scipy.stats import entropy

# Optional dependencies
try:
    from scipy.spatial import ConvexHull
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from shapely.geometry import Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

try:
    from pycocotools import mask as mask_utils
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("Warning: pycocotools not available. Mask extraction will be limited.")


# =============================================================================
# GLOBAL CONFIGURATION (V6)
# =============================================================================

CONFIG = {
    # Reporting Settings
    'CROPS_PER_FACE': 5,
    'PDF_FIGURE_SIZE': (8.5, 11),
    'CROP_PADDING': 30,
    'CROP_BG_COLOR': 'white',

    # V6: Quality Filtering
    'MIN_QUALITY_THRESHOLD': 0.15,
    'MAX_CANDIDATES_TO_SHOW': 20,

    # V6: Modern/Colorful Style - Orientation Colors
    'ORIENTATION_COLORS': {
        'front': '#FF4444',   # Red
        'back': '#44FF44',    # Green
        'left': '#4444FF',    # Blue
        'right': '#FFAA44',   # Orange
        'top': '#FF44FF',     # Magenta
        'bottom': '#44FFFF',  # Cyan (not typically used)
    },

    # Missing view indicator
    'MISSING_COLOR': '#CCCCCC',

    # Grade colors
    'GRADE_COLORS': {
        'A': '#2ECC71',  # Green
        'B': '#3498DB',  # Blue
        'C': '#F39C12',  # Orange
        'F': '#E74C3C',  # Red
    },

    # Grading Thresholds
    'THRESHOLDS': {
        'coverage_good': 0.15,
        'coverage_fair': 0.08,
        'occlusion_severe': 0.30,
        'diversity_good': 0.60,
        'diversity_fair': 0.40,
    },

    # Semantic face labels
    'VISIBLE_FACES': ['front', 'back', 'left', 'right', 'top'],
    'ALL_FACES': ['front', 'back', 'left', 'right', 'top', 'bottom'],
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GradeResult:
    letter: str
    score: float
    label: str
    description: str
    color: str


@dataclass
class CandidateFrame:
    """A candidate frame for user selection."""
    frame: str
    quality_score: float
    frame_idx: int
    crop_image: Optional[np.ndarray] = None
    approved: bool = False


# =============================================================================
# VIEWPOINT ANALYZER CLASS (V6)
# =============================================================================

class ViewpointAnalyzer:
    """Viewpoint analysis with interactive selection support."""

    def __init__(self, annotator_output_dir, images_dir=None):
        self.annotator_output_dir = Path(annotator_output_dir)
        self.images_dir = Path(images_dir) if images_dir else None

        self._validate_input_paths()

        self.all_bbox_data = self._load_annotated_bboxes()
        self.semantic_faces = self._load_semantic_faces()
        self.labeled_tracks = list(self.semantic_faces.keys())
        self.frame_order = self._determine_frame_order()

        self.semantic_face_colors = CONFIG['ORIENTATION_COLORS']

        print(f"Viewpoint Analyzer v6 initialized:")
        print(f"  Annotator output: {self.annotator_output_dir}")
        print(f"  Tracks with semantic labels: {self.labeled_tracks}")
        print(f"  Total frames: {len(self.frame_order)}")

    def _validate_input_paths(self):
        bbox_dir = self.annotator_output_dir / "bounding_boxes"
        semantic_file = (self.annotator_output_dir /
                        "corrected_labels" / "semantic_faces" / "manual_labels.json")

        errors = []
        if not bbox_dir.exists():
            errors.append(f"Bounding boxes directory not found: {bbox_dir}")
        elif not list(bbox_dir.glob("*.json")):
            errors.append(f"No bounding box JSON files in: {bbox_dir}")

        if not semantic_file.exists():
            errors.append(f"Semantic face labels not found: {semantic_file}")

        if errors:
            raise FileNotFoundError("\n".join(errors))

    def _load_annotated_bboxes(self):
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
        semantic_file = (self.annotator_output_dir /
                        "corrected_labels" / "semantic_faces" / "manual_labels.json")

        with open(semantic_file, 'r') as f:
            raw_labels = json.load(f)

        semantic_faces = {}

        for track_id_str, frames_data in raw_labels.items():
            track_id = int(track_id_str)
            if track_id not in self.all_bbox_data:
                continue

            semantic_faces[track_id] = {}

            for frame_name, label_to_index in frames_data.items():
                frame_key = str(frame_name)
                if frame_key not in self.all_bbox_data[track_id]:
                    continue

                bbox_data = self.all_bbox_data[track_id][frame_key]
                all_faces = self.get_all_faces_from_bbox(bbox_data)

                semantic_faces[track_id][frame_key] = {}

                for semantic_label, face_index in label_to_index.items():
                    face_key = f'f{face_index}'
                    if face_key in all_faces:
                        semantic_faces[track_id][frame_key][semantic_label] = all_faces[face_key]

                self._infer_opposite_faces(semantic_faces[track_id][frame_key], all_faces)

        total_labels = sum(len(frames) for frames in semantic_faces.values())
        print(f"  Loaded semantic labels: {len(semantic_faces)} tracks, {total_labels} frame-label sets")
        return semantic_faces

    def _determine_frame_order(self):
        all_frame_names = set()
        for track_data in self.all_bbox_data.values():
            all_frame_names.update(track_data.keys())

        def extract_numeric_part(frame_name):
            numbers = re.findall(r'\d+', str(frame_name))
            return int(numbers[0]) if numbers else float('inf')

        return sorted(list(all_frame_names), key=extract_numeric_part)

    def _load_camera_params(self, frame_name):
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
            return None

    # Geometry methods
    def get_bbox_corners(self, center, dimensions, rotation_matrix):
        l, w, h = dimensions
        corners_local = np.array([
            [-l/2, -w/2, -h/2], [+l/2, -w/2, -h/2],
            [+l/2, +w/2, -h/2], [-l/2, +w/2, -h/2],
            [-l/2, -w/2, +h/2], [+l/2, -w/2, +h/2],
            [+l/2, +w/2, +h/2], [-l/2, +w/2, +h/2],
        ])
        corners_world = (rotation_matrix @ corners_local.T).T + center
        return corners_world

    def compute_face_from_corners(self, corners, indices, box_center):
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
        return {'center': face_center, 'normal': normal, 'corners': face_corners, 'area': area}

    def get_all_faces_from_bbox(self, bbox_data):
        corners = self.get_bbox_corners(
            bbox_data['center'], bbox_data['dimensions'], bbox_data['rotation_matrix']
        )
        box_center = np.mean(corners, axis=0)
        face_indices = {
            'f0': [0, 1, 5, 4], 'f1': [2, 3, 7, 6], 'f2': [0, 3, 7, 4],
            'f3': [1, 2, 6, 5], 'f4': [4, 5, 6, 7], 'f5': [0, 1, 2, 3],
        }
        faces = {}
        for face_id, indices in face_indices.items():
            faces[face_id] = self.compute_face_from_corners(corners, indices, box_center)
        return faces

    def _infer_opposite_faces(self, semantic_assignments, all_faces):
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
                best_match_id = None
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

    # Visibility and quality methods
    def _calculate_face_visibility(self, face_data, camera_params):
        try:
            t = camera_params['t']
            camera_pos = t
            face_to_camera = camera_pos - face_data['center']
            face_to_camera = face_to_camera / np.linalg.norm(face_to_camera)
            visibility_score = np.dot(face_data['normal'], face_to_camera)
            return visibility_score > 0, visibility_score
        except:
            return False, 0.0

    def _project_face_to_2d(self, face_data, camera_params, img_shape):
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

    def _calculate_face_quality(self, face_data, camera_params, img_shape):
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

            quality_score = (
                0.25 * abs(visibility_score) +
                0.25 * area_score +
                0.25 * distance_score +
                0.25 * aspect_ratio
            )
            return quality_score
        except:
            return 0.0

    def compute_frame_qualities(self, track_id: int) -> Dict[str, Dict[str, float]]:
        """Compute quality scores for all frames and orientations."""
        if track_id not in self.semantic_faces:
            return {}

        semantic_labels = CONFIG['ALL_FACES']
        frame_quality_scores = {}

        sorted_frames = sorted(
            self.semantic_faces[track_id].keys(),
            key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0
        )

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
                    frame_qualities[label] = 0.0
                elif label in semantic_faces:
                    quality_score = self._calculate_face_quality(
                        semantic_faces[label], camera_params, img_shape
                    )
                    frame_qualities[label] = quality_score
                else:
                    frame_qualities[label] = 0.0

            frame_quality_scores[frame_name] = frame_qualities

        return frame_quality_scores

    def get_candidates_for_orientation(self, track_id: int, orientation: str,
                                       frame_qualities: Dict,
                                       min_quality: float = None,
                                       max_candidates: int = None) -> List[CandidateFrame]:
        if min_quality is None:
            min_quality = CONFIG['MIN_QUALITY_THRESHOLD']
        if max_candidates is None:
            max_candidates = CONFIG['MAX_CANDIDATES_TO_SHOW']

        candidates = []

        sorted_frames = sorted(
            frame_qualities.keys(),
            key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0
        )

        for frame_idx, frame_name in enumerate(sorted_frames):
            quality = frame_qualities[frame_name].get(orientation, 0.0)
            if quality >= min_quality:
                candidates.append(CandidateFrame(
                    frame=frame_name,
                    quality_score=quality,
                    frame_idx=frame_idx
                ))

        candidates.sort(key=lambda x: x.quality_score, reverse=True)
        return candidates[:max_candidates]

    def get_quality_statistics(self, track_id: int, frame_qualities: Dict) -> Dict:
        stats = {}
        for orientation in CONFIG['VISIBLE_FACES']:
            qualities = [
                frame_qualities[f].get(orientation, 0.0)
                for f in frame_qualities
            ]
            non_zero = [q for q in qualities if q > 0]

            stats[orientation] = {
                'total_frames': len(qualities),
                'frames_with_visibility': len(non_zero),
                'max_quality': max(qualities) if qualities else 0,
                'mean_quality': np.mean(non_zero) if non_zero else 0,
                'frames_above_0.1': sum(1 for q in qualities if q >= 0.1),
                'frames_above_0.2': sum(1 for q in qualities if q >= 0.2),
                'frames_above_0.3': sum(1 for q in qualities if q >= 0.3),
            }
        return stats


# =============================================================================
# MASK CROP EXTRACTOR
# =============================================================================

class MaskCropExtractor:
    """Extracts masked animal crops from images."""

    def __init__(self, images_dir: Path, mask_dir: Path, results_dir: Path):
        self.images_dir = Path(images_dir) if images_dir else None
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.results_dir = Path(results_dir) if results_dir else None
        self.mask_track_mapping = self._load_mask_track_mapping()

    def _load_mask_track_mapping(self) -> Dict:
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
            pass

        return None

    def _create_error_image(self, size: int = 200) -> np.ndarray:
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:, :, 2] = 80
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "IMG ERR"
        text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
        text_x = (size - text_size[0]) // 2
        text_y = (size + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), font, 0.6, (255, 255, 255), 2)
        return img

    def extract_crop(self, frame_name: str, track_id: int,
                    padding: int = None, background: str = None) -> Optional[np.ndarray]:
        if padding is None:
            padding = CONFIG['CROP_PADDING']
        if background is None:
            background = CONFIG['CROP_BG_COLOR']

        try:
            image = self._load_image(frame_name)
            if image is None:
                return self._create_error_image()

            mask = self._load_mask(frame_name, track_id)
            if mask is None:
                h, w = image.shape[:2]
                crop_size = min(h, w) // 2
                cx, cy = w // 2, h // 2
                x1 = max(0, cx - crop_size)
                x2 = min(w, cx + crop_size)
                y1 = max(0, cy - crop_size)
                y2 = min(h, cy + crop_size)
                return image[y1:y2, x1:x2].copy()

            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                return self._create_error_image()

            mask_x1, mask_x2 = xs.min(), xs.max()
            mask_y1, mask_y2 = ys.min(), ys.max()

            mask_w = mask_x2 - mask_x1
            mask_h = mask_y2 - mask_y1
            mask_cx = (mask_x1 + mask_x2) // 2
            mask_cy = (mask_y1 + mask_y2) // 2

            max_dim = max(mask_w, mask_h) + 2 * padding
            half_dim = max_dim // 2

            img_h, img_w = image.shape[:2]
            x1 = max(0, mask_cx - half_dim)
            x2 = min(img_w, mask_cx + half_dim)
            y1 = max(0, mask_cy - half_dim)
            y2 = min(img_h, mask_cy + half_dim)

            cropped_image = image[y1:y2, x1:x2].copy()
            cropped_mask = mask[y1:y2, x1:x2]

            if background == 'white':
                result = np.ones_like(cropped_image) * 255
                result[cropped_mask > 0] = cropped_image[cropped_mask > 0]
            else:
                result = cropped_image

            return result

        except Exception as e:
            return self._create_error_image()


# =============================================================================
# INTERACTIVE FRAME SELECTOR
# =============================================================================

class InteractiveFrameSelector:
    """Interactive GUI for selecting best frames per orientation."""

    def __init__(self, analyzer: ViewpointAnalyzer,
                 crop_extractor: MaskCropExtractor,
                 track_id: int):
        self.analyzer = analyzer
        self.crop_extractor = crop_extractor
        self.track_id = track_id
        self.selections = {}

    def run_selection(self, min_quality: float = None) -> Dict[str, List[str]]:
        if min_quality is None:
            min_quality = CONFIG['MIN_QUALITY_THRESHOLD']

        print(f"\n{'='*70}")
        print(f"INTERACTIVE FRAME SELECTION - Track {self.track_id}")
        print(f"{'='*70}")
        print(f"  Min quality threshold: {min_quality:.2f}")
        print(f"  Max candidates to show: {CONFIG['MAX_CANDIDATES_TO_SHOW']}")

        frame_qualities = self.analyzer.compute_frame_qualities(self.track_id)

        stats = self.analyzer.get_quality_statistics(self.track_id, frame_qualities)
        print(f"\n  Quality Statistics:")
        for orient, s in stats.items():
            print(f"    {orient:6s}: {s['frames_with_visibility']:3d} visible, "
                  f"max={s['max_quality']:.2f}, "
                  f">=0.2: {s['frames_above_0.2']}, >=0.3: {s['frames_above_0.3']}")

        visible_labels = CONFIG['VISIBLE_FACES']
        self.selections = {label: [] for label in visible_labels}

        for orientation in visible_labels:
            candidates = self.analyzer.get_candidates_for_orientation(
                self.track_id, orientation, frame_qualities, min_quality
            )

            if not candidates:
                print(f"\n  [{orientation.upper()}] No candidates above threshold {min_quality:.2f}")
                print(f"    (max quality for {orientation}: {stats[orientation]['max_quality']:.3f})")
                continue

            print(f"\n  [{orientation.upper()}] Showing {len(candidates)} candidates (sorted by quality)")

            for candidate in candidates:
                if self.crop_extractor:
                    candidate.crop_image = self.crop_extractor.extract_crop(
                        candidate.frame, self.track_id
                    )

            selected = self._show_selection_gui(orientation, candidates)
            self.selections[orientation] = selected

            print(f"    Selected {len(selected)} frames: {selected}")

        return self.selections, frame_qualities

    def _show_selection_gui(self, orientation: str,
                           candidates: List[CandidateFrame]) -> List[str]:
        n_candidates = len(candidates)
        if n_candidates == 0:
            return []

        n_cols = min(5, n_candidates)
        n_rows = (n_candidates + n_cols - 1) // n_cols

        fig_width = 3 * n_cols
        fig_height = 3.5 * n_rows + 1.5

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        fig.suptitle(f'Select frames for {orientation.upper()} view (Track {self.track_id})\n'
                    f'Click to toggle selection (green=selected), close window when done\n'
                    f'Showing {n_candidates} candidates sorted by quality',
                    fontsize=12, fontweight='bold')

        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        selection_state = [False] * n_candidates
        border_rects = []

        def update_borders():
            for i, (rect, selected) in enumerate(zip(border_rects, selection_state)):
                if selected:
                    rect.set_edgecolor('lime')
                    rect.set_linewidth(6)
                else:
                    rect.set_edgecolor('gray')
                    rect.set_linewidth(1)
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes is None:
                return
            for i in range(n_candidates):
                row = i // n_cols
                col = i % n_cols
                if event.inaxes == axes[row, col]:
                    selection_state[i] = not selection_state[i]
                    update_borders()
                    break

        for i, candidate in enumerate(candidates):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            if candidate.crop_image is not None:
                img = candidate.crop_image
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            else:
                ax.set_facecolor('#E0E0E0')
                ax.text(0.5, 0.5, 'No image', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)

            frame_num = re.findall(r'\d+', str(candidate.frame))
            frame_str = frame_num[0] if frame_num else candidate.frame
            ax.set_title(f'F:{frame_str} Q:{candidate.quality_score:.3f}', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

            rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                 fill=False, edgecolor='gray', linewidth=1)
            ax.add_patch(rect)
            border_rects.append(rect)

        for i in range(n_candidates, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')

        fig.canvas.mpl_connect('button_press_event', on_click)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

        selected_frames = [
            candidates[i].frame for i in range(n_candidates) if selection_state[i]
        ]

        return selected_frames


# =============================================================================
# QUALITY GRADER
# =============================================================================

class QualityGrader:
    GRADE_COLORS = CONFIG['GRADE_COLORS']
    GRADE_LABELS = {'A': 'Excellent', 'B': 'Good', 'C': 'Fair', 'F': 'Poor'}

    def grade_selections(self, selections: Dict[str, List[str]]) -> GradeResult:
        visible_labels = CONFIG['VISIBLE_FACES']

        orientations_with_data = sum(1 for label in visible_labels
                                     if len(selections.get(label, [])) > 0)

        total_selected = sum(len(selections.get(label, [])) for label in visible_labels)

        if orientations_with_data >= 5:
            letter = 'A'
            desc = f"All 5 orientations have selected frames ({total_selected} total)"
        elif orientations_with_data >= 4:
            letter = 'B'
            desc = f"{orientations_with_data}/5 orientations covered ({total_selected} total)"
        elif orientations_with_data >= 3:
            letter = 'C'
            desc = f"Only {orientations_with_data}/5 orientations ({total_selected} total)"
        else:
            letter = 'F'
            desc = f"Poor: only {orientations_with_data}/5 orientations ({total_selected} total)"

        missing = [label for label in visible_labels if len(selections.get(label, [])) == 0]
        if missing:
            desc += f"\nMissing: {', '.join(missing)}"

        return GradeResult(
            letter=letter,
            score=orientations_with_data / 5.0,
            label=self.GRADE_LABELS[letter],
            description=desc,
            color=self.GRADE_COLORS[letter]
        )


# =============================================================================
# AGGREGATE REPORT GENERATOR (V6 - NEW)
# =============================================================================

class AggregateReportGenerator:
    """Generates publication-quality aggregate PDF combining all tracklets."""

    def __init__(self, crop_extractor: MaskCropExtractor = None, video_name: str = ""):
        self.crop_extractor = crop_extractor
        self.video_name = video_name
        self.grader = QualityGrader()
        self.orientation_colors = CONFIG['ORIENTATION_COLORS']
        self.missing_color = CONFIG['MISSING_COLOR']

    def load_all_saved_selections(self, viewpoint_dir: Path) -> Dict[int, Dict[str, List[str]]]:
        """Load all approved_selections_track*.json files."""
        all_results = {}
        for f in sorted(viewpoint_dir.glob("approved_selections_track*.json")):
            with open(f, 'r') as fp:
                data = json.load(fp)
                track_id = data.get('track_id')
                if track_id is not None:
                    # Handle both formats
                    if 'selections' in data:
                        all_results[track_id] = data['selections']
                    else:
                        all_results[track_id] = {k: v for k, v in data.items()
                                                  if k in CONFIG['VISIBLE_FACES']}
        return all_results

    def generate_aggregate_report(self, all_results: Dict[int, Dict[str, List[str]]],
                                  output_path: Path) -> Path:
        """Generate aggregate PDF with coverage matrix and compact filmstrips."""
        output_path.parent.mkdir(exist_ok=True)

        print(f"\n{'='*70}")
        print("GENERATING AGGREGATE PDF (V6 - Publication Quality)")
        print(f"{'='*70}")
        print(f"  Tracks: {list(all_results.keys())}")
        print(f"  Output: {output_path}")

        # Compute grades for all tracks
        grades = {track_id: self.grader.grade_selections(selections)
                  for track_id, selections in all_results.items()}

        with PdfPages(str(output_path)) as pdf:
            # Page 1: Coverage matrix with statistics
            self._create_coverage_matrix_page(pdf, all_results, grades)

            # Page 2+: Compact filmstrips
            self._create_compact_filmstrip_pages(pdf, all_results, grades)

        print(f"\nSaved aggregate PDF to: {output_path}")
        return output_path

    def _create_coverage_matrix_page(self, pdf: PdfPages,
                                     all_results: Dict[int, Dict[str, List[str]]],
                                     grades: Dict[int, GradeResult]):
        """Create coverage matrix page with modern/colorful style."""
        fig = plt.figure(figsize=(8.5, 11))

        # Title with gradient-like background
        fig.suptitle(f'VIEWPOINT COVERAGE SUMMARY\n{self.video_name}',
                    fontsize=16, fontweight='bold', y=0.95)

        # Create main axes for the matrix
        ax_matrix = fig.add_axes([0.1, 0.35, 0.8, 0.5])
        ax_matrix.axis('off')

        visible_labels = CONFIG['VISIBLE_FACES']
        track_ids = sorted(all_results.keys())
        n_tracks = len(track_ids)
        n_cols = len(visible_labels) + 2  # +1 for track label, +1 for grade

        # Cell dimensions
        cell_width = 0.85 / n_cols
        cell_height = 0.9 / (n_tracks + 1)  # +1 for header

        # Draw header row
        header_y = 1.0 - cell_height

        # Track column header
        ax_matrix.text(cell_width / 2, header_y + cell_height / 2, 'Track',
                      ha='center', va='center', fontsize=10, fontweight='bold')

        # Orientation column headers with colors
        for col_idx, label in enumerate(visible_labels):
            x = (col_idx + 1) * cell_width + cell_width / 2
            color = self.orientation_colors[label]

            # Colored background
            rect = FancyBboxPatch((x - cell_width/2 + 0.01, header_y + 0.01),
                                  cell_width - 0.02, cell_height - 0.02,
                                  boxstyle="round,pad=0.02",
                                  facecolor=color, alpha=0.3, edgecolor=color,
                                  linewidth=2, transform=ax_matrix.transAxes)
            ax_matrix.add_patch(rect)

            ax_matrix.text(x, header_y + cell_height / 2, label.upper(),
                          ha='center', va='center', fontsize=9, fontweight='bold')

        # Grade column header
        grade_x = (len(visible_labels) + 1) * cell_width + cell_width / 2
        ax_matrix.text(grade_x, header_y + cell_height / 2, 'Grade',
                      ha='center', va='center', fontsize=10, fontweight='bold')

        # Draw data rows
        for row_idx, track_id in enumerate(track_ids):
            selections = all_results[track_id]
            grade = grades[track_id]
            y = header_y - (row_idx + 1) * cell_height

            # Track ID
            ax_matrix.text(cell_width / 2, y + cell_height / 2, f'Track {track_id}',
                          ha='center', va='center', fontsize=9, fontweight='bold')

            # Orientation cells
            for col_idx, label in enumerate(visible_labels):
                x = (col_idx + 1) * cell_width + cell_width / 2
                count = len(selections.get(label, []))

                if count > 0:
                    # Has views - colored cell
                    color = self.orientation_colors[label]
                    rect = FancyBboxPatch((x - cell_width/2 + 0.01, y + 0.01),
                                          cell_width - 0.02, cell_height - 0.02,
                                          boxstyle="round,pad=0.02",
                                          facecolor=color, alpha=0.4,
                                          edgecolor=color, linewidth=1,
                                          transform=ax_matrix.transAxes)
                    ax_matrix.add_patch(rect)
                    ax_matrix.text(x, y + cell_height / 2, str(count),
                                  ha='center', va='center', fontsize=11, fontweight='bold')
                else:
                    # Missing - gray with X
                    rect = FancyBboxPatch((x - cell_width/2 + 0.01, y + 0.01),
                                          cell_width - 0.02, cell_height - 0.02,
                                          boxstyle="round,pad=0.02",
                                          facecolor=self.missing_color, alpha=0.3,
                                          edgecolor=self.missing_color, linewidth=1,
                                          transform=ax_matrix.transAxes)
                    ax_matrix.add_patch(rect)
                    ax_matrix.text(x, y + cell_height / 2, 'X',
                                  ha='center', va='center', fontsize=11,
                                  color='#666666', fontweight='bold')

            # Grade cell
            grade_color = grade.color
            rect = FancyBboxPatch((grade_x - cell_width/2 + 0.01, y + 0.01),
                                  cell_width - 0.02, cell_height - 0.02,
                                  boxstyle="round,pad=0.02",
                                  facecolor=grade_color, alpha=0.4,
                                  edgecolor=grade_color, linewidth=2,
                                  transform=ax_matrix.transAxes)
            ax_matrix.add_patch(rect)
            ax_matrix.text(grade_x, y + cell_height / 2, grade.letter,
                          ha='center', va='center', fontsize=12, fontweight='bold')

        # Statistics section
        ax_stats = fig.add_axes([0.1, 0.08, 0.8, 0.2])
        ax_stats.axis('off')

        # Calculate statistics
        total_frames = sum(sum(len(v) for v in sel.values()) for sel in all_results.values())
        orientations_used = set()
        for sel in all_results.values():
            for label, frames in sel.items():
                if len(frames) > 0:
                    orientations_used.add(label)

        # Grade distribution
        grade_counts = Counter(g.letter for g in grades.values())

        stats_text = (
            f"Total Tracklets: {n_tracks}    |    "
            f"Total Frames Selected: {total_frames}    |    "
            f"Orientations Used: {len(orientations_used)}/5\n\n"
            f"Grade Distribution:  "
            f"A: {grade_counts.get('A', 0)}  |  "
            f"B: {grade_counts.get('B', 0)}  |  "
            f"C: {grade_counts.get('C', 0)}  |  "
            f"F: {grade_counts.get('F', 0)}"
        )

        ax_stats.text(0.5, 0.5, stats_text,
                     ha='center', va='center', fontsize=11,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0',
                              edgecolor='#CCCCCC', linewidth=1))

        # Legend for orientation colors
        legend_y = 0.02
        legend_x_start = 0.15
        for i, label in enumerate(visible_labels):
            color = self.orientation_colors[label]
            x = legend_x_start + i * 0.15
            rect = FancyBboxPatch((x, legend_y), 0.03, 0.03,
                                  boxstyle="round,pad=0.01",
                                  facecolor=color, alpha=0.5,
                                  transform=fig.transFigure)
            fig.patches.append(rect)
            fig.text(x + 0.04, legend_y + 0.015, label.capitalize(),
                    fontsize=8, va='center')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_compact_filmstrip_pages(self, pdf: PdfPages,
                                        all_results: Dict[int, Dict[str, List[str]]],
                                        grades: Dict[int, GradeResult]):
        """Create compact horizontal filmstrip pages."""
        track_ids = sorted(all_results.keys())
        visible_labels = CONFIG['VISIBLE_FACES']

        # Calculate how many tracks fit per page (approximately 3-4)
        tracks_per_page = 3

        for page_start in range(0, len(track_ids), tracks_per_page):
            page_tracks = track_ids[page_start:page_start + tracks_per_page]

            # Calculate figure height based on content
            max_frames_per_track = []
            for track_id in page_tracks:
                selections = all_results[track_id]
                total = sum(len(v) for v in selections.values())
                max_frames_per_track.append(max(total, 1))

            fig = plt.figure(figsize=(8.5, 11))

            # Title
            page_num = page_start // tracks_per_page + 1
            total_pages = (len(track_ids) + tracks_per_page - 1) // tracks_per_page
            fig.suptitle(f'Viewpoint Filmstrips - Page {page_num}/{total_pages}',
                        fontsize=14, fontweight='bold', y=0.98)

            # Divide page into track sections
            n_page_tracks = len(page_tracks)
            track_height = 0.85 / n_page_tracks

            for track_idx, track_id in enumerate(page_tracks):
                selections = all_results[track_id]
                grade = grades[track_id]

                # Track section position
                track_top = 0.92 - track_idx * track_height
                track_bottom = track_top - track_height + 0.02

                self._draw_track_strip(fig, track_id, selections, grade,
                                       track_top, track_bottom)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    def _draw_track_strip(self, fig, track_id: int, selections: Dict[str, List[str]],
                          grade: GradeResult, top: float, bottom: float):
        """Draw a single track's horizontal filmstrip."""
        visible_labels = CONFIG['VISIBLE_FACES']

        # Collect all frames with their orientations
        all_frames = []
        for label in visible_labels:
            for frame in selections.get(label, []):
                all_frames.append((label, frame))

        # Track header bar with gradient
        header_height = 0.025
        header_ax = fig.add_axes([0.05, top - header_height, 0.9, header_height])
        header_ax.axis('off')

        # Gradient background for header
        gradient = np.linspace(0, 1, 100).reshape(1, -1)
        header_ax.imshow(gradient, aspect='auto', cmap='Blues', alpha=0.3,
                        extent=[0, 1, 0, 1])

        header_ax.text(0.02, 0.5, f'TRACK {track_id}',
                      ha='left', va='center', fontsize=11, fontweight='bold')

        # Grade badge
        grade_color = grade.color
        header_ax.text(0.98, 0.5, f'Grade: {grade.letter}',
                      ha='right', va='center', fontsize=10, fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor=grade_color,
                               alpha=0.5, edgecolor=grade_color))

        # Filmstrip area
        strip_top = top - header_height - 0.01
        strip_height = (strip_top - bottom) * 0.7
        strip_ax = fig.add_axes([0.05, strip_top - strip_height, 0.9, strip_height])
        strip_ax.axis('off')

        if len(all_frames) == 0:
            strip_ax.text(0.5, 0.5, 'No frames selected',
                         ha='center', va='center', fontsize=10, color='gray',
                         style='italic')
        else:
            # Draw frames horizontally
            n_frames = len(all_frames)
            frame_width = min(0.15, 0.9 / n_frames)

            for i, (label, frame_name) in enumerate(all_frames):
                x_pos = 0.02 + i * (frame_width + 0.01)

                # Frame background with orientation color
                color = self.orientation_colors[label]

                # Try to load crop image
                crop_img = None
                if self.crop_extractor:
                    crop_img = self.crop_extractor.extract_crop(frame_name, track_id)

                # Create mini axes for each frame
                frame_ax = fig.add_axes([0.05 + x_pos, strip_top - strip_height,
                                         frame_width * 0.9, strip_height * 0.9])

                if crop_img is not None:
                    if len(crop_img.shape) == 3 and crop_img.shape[2] == 3:
                        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    frame_ax.imshow(crop_img)
                else:
                    frame_ax.set_facecolor('#EEEEEE')
                    frame_ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                                 fontsize=8, color='gray')

                frame_ax.axis('off')

                # Color-coded border
                for spine in frame_ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)

                # Orientation badge
                frame_ax.text(0.5, -0.08, label.upper()[:3],
                             ha='center', va='top', fontsize=7, fontweight='bold',
                             transform=frame_ax.transAxes,
                             bbox=dict(boxstyle='round,pad=0.15', facecolor=color,
                                      alpha=0.7, edgecolor='none'))

                # Frame number
                frame_num = re.findall(r'\d+', str(frame_name))
                frame_str = frame_num[0] if frame_num else frame_name
                frame_ax.text(0.5, 1.05, f'F:{frame_str}',
                             ha='center', va='bottom', fontsize=6,
                             transform=frame_ax.transAxes)

        # Missing views indicator
        missing_labels = [label for label in visible_labels
                         if len(selections.get(label, [])) == 0]

        if missing_labels:
            missing_y = bottom + 0.01
            missing_ax = fig.add_axes([0.05, missing_y, 0.9, 0.02])
            missing_ax.axis('off')

            missing_text = "Missing: "
            for i, label in enumerate(missing_labels):
                color = self.orientation_colors[label]
                x_pos = 0.08 + i * 0.12

                # Small colored badge with X
                missing_ax.add_patch(FancyBboxPatch(
                    (x_pos, 0.2), 0.08, 0.6,
                    boxstyle="round,pad=0.1",
                    facecolor=self.missing_color, alpha=0.5,
                    edgecolor=color, linewidth=1,
                    transform=missing_ax.transAxes
                ))
                missing_ax.text(x_pos + 0.04, 0.5, f'{label[:3].upper()}',
                               ha='center', va='center', fontsize=6,
                               color='#666666', fontweight='bold',
                               transform=missing_ax.transAxes)

            missing_ax.text(0.02, 0.5, 'Missing:',
                           ha='left', va='center', fontsize=7,
                           color='#666666', transform=missing_ax.transAxes)


# =============================================================================
# PDF REPORT GENERATOR (V6 - Per-track, kept for compatibility)
# =============================================================================

class PDFReportGenerator:
    """Generates filmstrip PDF with vertical strip layout (per-track)."""

    def __init__(self, analyzer: ViewpointAnalyzer,
                 crop_extractor: MaskCropExtractor = None):
        self.analyzer = analyzer
        self.crop_extractor = crop_extractor
        self.grader = QualityGrader()

    def generate_report(self, track_id: int, selections: Dict[str, List[str]],
                       frame_qualities: Dict,
                       output_path: Path = None) -> Path:
        if output_path is None:
            output_path = (self.analyzer.annotator_output_dir /
                          "viewpoint_analysis" / f"filmstrip_v6_track{track_id}.pdf")

        output_path.parent.mkdir(exist_ok=True)

        print(f"\n{'='*70}")
        print("GENERATING FILMSTRIP PDF (V6)")
        print(f"{'='*70}")

        grade = self.grader.grade_selections(selections)

        with PdfPages(str(output_path)) as pdf:
            self._create_vertical_filmstrip_page(pdf, track_id, selections, frame_qualities, grade)

        print(f"\nSaved PDF report to: {output_path}")
        return output_path

    def _create_vertical_filmstrip_page(self, pdf: PdfPages, track_id: int,
                                        selections: Dict[str, List[str]],
                                        frame_qualities: Dict,
                                        grade: GradeResult):
        visible_labels = CONFIG['VISIBLE_FACES']
        n_cols = len(visible_labels)

        max_frames = max(len(selections.get(l, [])) for l in visible_labels)
        max_frames = max(max_frames, 1)

        fig = plt.figure(figsize=(11, 8.5))

        n_rows = max_frames + 1
        gs = GridSpec(n_rows, n_cols, figure=fig,
                     height_ratios=[0.15] + [0.85 / max_frames] * max_frames,
                     left=0.02, right=0.98, top=0.92, bottom=0.02,
                     hspace=0.08, wspace=0.05)

        total_selected = sum(len(selections.get(l, [])) for l in visible_labels)
        fig.suptitle(f'Track {track_id} - Viewpoint Filmstrip (V6)\n'
                    f'Grade: {grade.letter} | {total_selected} frames selected',
                    fontsize=14, fontweight='bold')

        for col_idx, label in enumerate(visible_labels):
            ax_header = fig.add_subplot(gs[0, col_idx])
            ax_header.axis('off')

            n_selected = len(selections.get(label, []))
            color = CONFIG['ORIENTATION_COLORS'][label] if n_selected > 0 else CONFIG['MISSING_COLOR']

            ax_header.text(0.5, 0.5, f'{label.upper()}\n({n_selected})',
                          ha='center', va='center',
                          fontsize=11, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor=color, alpha=0.4))

        for col_idx, label in enumerate(visible_labels):
            selected_frames = selections.get(label, [])
            orientation_color = CONFIG['ORIENTATION_COLORS'][label]

            for row_idx in range(max_frames):
                ax = fig.add_subplot(gs[row_idx + 1, col_idx])

                if row_idx < len(selected_frames):
                    frame_name = selected_frames[row_idx]
                    quality = frame_qualities.get(frame_name, {}).get(label, 0)

                    crop_image = None
                    if self.crop_extractor:
                        crop_image = self.crop_extractor.extract_crop(frame_name, track_id)

                    if crop_image is not None:
                        if crop_image.shape[-1] == 3:
                            crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
                        ax.imshow(crop_image)

                        frame_num = re.findall(r'\d+', str(frame_name))
                        frame_str = frame_num[0] if frame_num else frame_name
                        ax.text(0.02, 0.98, f'F:{frame_str}',
                               transform=ax.transAxes, fontsize=7,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round,pad=0.1',
                                        facecolor='white', alpha=0.8))
                        ax.text(0.02, 0.02, f'Q:{quality:.2f}',
                               transform=ax.transAxes, fontsize=7,
                               verticalalignment='bottom',
                               bbox=dict(boxstyle='round,pad=0.1',
                                        facecolor='white', alpha=0.8))

                        for spine in ax.spines.values():
                            spine.set_edgecolor(orientation_color)
                            spine.set_linewidth(3)
                    else:
                        ax.set_facecolor('#EEEEEE')
                        ax.text(0.5, 0.5, 'No image', ha='center', va='center',
                               fontsize=8, color='gray')
                else:
                    ax.set_facecolor('#F8F8F8')

                ax.set_xticks([])
                ax.set_yticks([])

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def save_json_report(self, track_id: int, selections: Dict[str, List[str]],
                        frame_qualities: Dict,
                        output_path: Path = None) -> Path:
        if output_path is None:
            output_path = (self.analyzer.annotator_output_dir /
                          "viewpoint_analysis" / f"selections_v6_track{track_id}.json")

        output_path.parent.mkdir(exist_ok=True)

        grade = self.grader.grade_selections(selections)

        report = {
            'track_id': track_id,
            'timestamp': datetime.now().isoformat(),
            'version': 'v6',
            'grade': {
                'letter': grade.letter,
                'score': grade.score,
                'description': grade.description
            },
            'selections': selections,
            'frame_qualities': {
                frame: {k: float(v) for k, v in quals.items()}
                for frame, quals in frame_qualities.items()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Saved JSON report to: {output_path}")
        return output_path


# =============================================================================
# MAIN ENTRY POINT (V6)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Viewpoint Analyzer v6 - Publication-Quality Aggregate PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate aggregate PDF from saved selections (recommended)
    python viewpoint_analyzer_v6.py --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/ --load_saved --aggregate

    # Interactive selection + aggregate PDF
    python viewpoint_analyzer_v6.py --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/ --aggregate

    # Process specific tracks with aggregate
    python viewpoint_analyzer_v6.py --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/ --track_id 0 1 5 --aggregate
        """
    )

    parser.add_argument("--annotator_output", required=True,
                       help="Path to annotator tool output directory")
    parser.add_argument("--images_dir", default=None,
                       help="Path to original images")
    parser.add_argument("--mask_dir", default=None,
                       help="Path to grounded-SAM mask directory")
    parser.add_argument("--results_dir", default=None,
                       help="Path to results directory")
    parser.add_argument("--min_quality", type=float, default=None,
                       help=f"Minimum quality threshold (default: {CONFIG['MIN_QUALITY_THRESHOLD']})")
    parser.add_argument("--max_candidates", type=int, default=None,
                       help=f"Max candidates per orientation (default: {CONFIG['MAX_CANDIDATES_TO_SHOW']})")
    parser.add_argument("--use_saved_selections", action="store_true",
                       help="Use previously saved selections instead of interactive mode")
    parser.add_argument("--load_saved", action="store_true",
                       help="Load from existing selection files (alias for --use_saved_selections)")
    parser.add_argument("--aggregate", action="store_true",
                       help="Generate aggregate PDF combining all tracks")
    parser.add_argument("--track_id", type=int, nargs='*', default=None,
                       help="Specific track ID(s) to process")
    parser.add_argument("--select_tracks", action="store_true",
                       help="Interactively select which tracks to process")
    parser.add_argument("--video_name", default=None,
                       help="Video name for PDF title (auto-detected if not provided)")

    args = parser.parse_args()

    # Handle alias
    if args.load_saved:
        args.use_saved_selections = True

    # Update CONFIG
    if args.min_quality is not None:
        CONFIG['MIN_QUALITY_THRESHOLD'] = args.min_quality
    if args.max_candidates is not None:
        CONFIG['MAX_CANDIDATES_TO_SHOW'] = args.max_candidates

    print("=" * 70)
    print("VIEWPOINT ANALYZER v6")
    print("Publication-Quality Aggregate PDF")
    print("=" * 70)
    print(f"  Min quality threshold: {CONFIG['MIN_QUALITY_THRESHOLD']}")
    print(f"  Aggregate mode: {args.aggregate}")

    # Auto-detect video name
    video_name = args.video_name
    if video_name is None:
        # Try to extract from path
        output_path = Path(args.annotator_output)
        if output_path.name == "corrected_bboxes":
            video_name = output_path.parent.name
        else:
            video_name = output_path.name
    print(f"  Video name: {video_name}")

    try:
        analyzer = ViewpointAnalyzer(
            annotator_output_dir=args.annotator_output,
            images_dir=args.images_dir
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return 1

    # Set up mask crop extractor
    crop_extractor = None
    if args.images_dir:
        mask_dir = args.mask_dir
        if mask_dir is None:
            possible_dirs = [
                Path(args.images_dir) / "grounded-sam",
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

    # Determine which tracks to process
    tracks_to_process = []
    viewpoint_dir = Path(args.annotator_output) / "viewpoint_analysis"

    if args.use_saved_selections and viewpoint_dir.exists():
        # Load from saved selections
        agg_gen = AggregateReportGenerator(crop_extractor, video_name)
        all_results = agg_gen.load_all_saved_selections(viewpoint_dir)

        if args.track_id:
            # Filter to specified tracks
            all_results = {k: v for k, v in all_results.items() if k in args.track_id}

        tracks_to_process = list(all_results.keys())
        print(f"\n  Loaded saved selections for {len(tracks_to_process)} track(s): {tracks_to_process}")
    else:
        # Interactive or specified tracks
        if args.select_tracks or (args.track_id is not None and len(args.track_id) == 0):
            print(f"\n  Available tracks with semantic labels: {analyzer.labeled_tracks}")
            print("  Select tracks to process (comma-separated, or 'all' for all tracks):")
            user_input = input("  > ").strip()

            if user_input.lower() == 'all' or user_input == '':
                tracks_to_process = analyzer.labeled_tracks
            else:
                try:
                    tracks_to_process = [int(t.strip()) for t in user_input.split(',')]
                    invalid = [t for t in tracks_to_process if t not in analyzer.labeled_tracks]
                    if invalid:
                        print(f"  WARNING: Tracks {invalid} don't have semantic labels, skipping them")
                        tracks_to_process = [t for t in tracks_to_process if t in analyzer.labeled_tracks]
                except ValueError:
                    print("  Invalid input. Processing all tracks.")
                    tracks_to_process = analyzer.labeled_tracks
        elif args.track_id is not None and len(args.track_id) > 0:
            tracks_to_process = args.track_id
            invalid = [t for t in tracks_to_process if t not in analyzer.labeled_tracks]
            if invalid:
                print(f"  WARNING: Tracks {invalid} don't have semantic labels, skipping them")
                tracks_to_process = [t for t in tracks_to_process if t in analyzer.labeled_tracks]
        else:
            tracks_to_process = analyzer.labeled_tracks

        all_results = {}

    if not tracks_to_process:
        print("\nERROR: No valid tracks to process")
        return 1

    print(f"\n  Will process {len(tracks_to_process)} track(s): {tracks_to_process}")

    # Process each track (if not loaded from saved)
    if not args.use_saved_selections or not all_results:
        all_results = {}
        for track_idx, track_id in enumerate(tracks_to_process):
            print(f"\n{'#'*70}")
            print(f"# PROCESSING TRACK {track_id} ({track_idx + 1}/{len(tracks_to_process)})")
            print(f"{'#'*70}")

            selection_file = (Path(args.annotator_output) / "viewpoint_analysis" /
                             f"approved_selections_track{track_id}.json")

            if args.use_saved_selections and selection_file.exists():
                print(f"\n  Loading saved selections from: {selection_file}")
                with open(selection_file, 'r') as f:
                    saved_data = json.load(f)
                selections = saved_data.get('selections', saved_data)
                frame_qualities = analyzer.compute_frame_qualities(track_id)
            else:
                selector = InteractiveFrameSelector(analyzer, crop_extractor, track_id)
                selections, frame_qualities = selector.run_selection(CONFIG['MIN_QUALITY_THRESHOLD'])

                selection_file.parent.mkdir(exist_ok=True)
                save_data = {
                    'track_id': track_id,
                    'timestamp': datetime.now().isoformat(),
                    'min_quality_threshold': CONFIG['MIN_QUALITY_THRESHOLD'],
                    'selections': selections
                }
                with open(selection_file, 'w') as f:
                    json.dump(save_data, f, indent=2)
                print(f"\n  Saved selections to: {selection_file}")

            print(f"\n{'='*70}")
            print(f"SELECTION SUMMARY - Track {track_id}")
            print(f"{'='*70}")
            for label in CONFIG['VISIBLE_FACES']:
                frames = selections.get(label, [])
                print(f"  {label.upper():6s}: {len(frames)} frames - {frames}")

            if crop_extractor and not args.aggregate:
                report_gen = PDFReportGenerator(analyzer, crop_extractor)
                report_gen.generate_report(track_id, selections, frame_qualities)
                report_gen.save_json_report(track_id, selections, frame_qualities)

            all_results[track_id] = selections

    # Generate aggregate PDF if requested
    if args.aggregate:
        agg_gen = AggregateReportGenerator(crop_extractor, video_name)
        output_path = viewpoint_dir / f"aggregate_filmstrip_v6_{video_name}.pdf"
        agg_gen.generate_aggregate_report(all_results, output_path)

    # Final summary
    print(f"\n{'='*70}")
    print(f"ALL TRACKS COMPLETE - Processed {len(tracks_to_process)} tracks")
    print(f"{'='*70}")
    for track_id, selections in all_results.items():
        total_frames = sum(len(v) for v in selections.values())
        orientations_covered = sum(1 for v in selections.values() if len(v) > 0)
        print(f"  Track {track_id}: {total_frames} frames selected, {orientations_covered}/5 orientations")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
