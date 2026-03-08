#!/usr/bin/env python3
"""
Exemplar-based Viewpoint Analyzer

Generates filmstrip output WITHOUT semantic labels. User selects exemplar
frames and assigns viewpoint labels. The system auto-classifies remaining
frames by viewing-angle similarity (camera-to-bbox direction in bbox-local frame).

Workflow:
    1. Generate a numbered contact sheet of all crops for a track
    2. User selects exemplar frames and assigns viewpoint labels (terminal or interactive)
    3. System classifies all frames by nearest-exemplar viewing angle
    4. Review classified contact sheet
    5. Generate filmstrip PDF

Usage:
    # Terminal-based selection (robust, works without display)
    python viewpoint_analyzer_exemplar.py \
        --results_dir results/paper_final/thursday/rhinos_cami/rhin-32_2 \
        --images_dir examples/wd_data/rhinos_cami/rhin-32_2 \
        --track_id 0

    # Interactive matplotlib selection
    python viewpoint_analyzer_exemplar.py \
        --results_dir results/paper_final/thursday/rhinos_cami/rhin-32_2 \
        --images_dir examples/wd_data/rhinos_cami/rhin-32_2 \
        --track_id 0 --interactive

    # Load previously saved exemplar selections
    python viewpoint_analyzer_exemplar.py \
        --results_dir results/paper_final/thursday/rhinos_cami/rhin-32_2 \
        --images_dir examples/wd_data/rhinos_cami/rhin-32_2 \
        --track_id 0 --load_saved

    # Adjust classification threshold (default 30 degrees)
    python viewpoint_analyzer_exemplar.py \
        --results_dir results/paper_final/thursday/rhinos_cami/rhin-32_2 \
        --images_dir examples/wd_data/rhinos_cami/rhin-32_2 \
        --track_id 0 --max_angle 45
"""

import os
import json
import re
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

try:
    from pycocotools import mask as mask_utils
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("Warning: pycocotools not available. Mask extraction will be limited.")


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'CROP_PADDING': 30,
    'CROP_BG_COLOR': 'white',
    'MAX_ANGLE_THRESHOLD': 30.0,   # degrees
    'MAX_FRAMES_PER_ROW': 10,
    'GALLERY_COLS': 10,

    'TYPOGRAPHY': {
        'title': 16,
        'subtitle': 13,
        'heading': 11,
        'body': 9,
        'caption': 7,
        'micro': 6,
    },

    # Color palette for dynamically-assigned viewpoint labels
    'VIEWPOINT_COLORS': [
        '#E53935',  # Red
        '#1E88E5',  # Blue
        '#43A047',  # Green
        '#FB8C00',  # Orange
        '#8E24AA',  # Purple
        '#00ACC1',  # Cyan
        '#F4511E',  # Deep Orange
        '#3949AB',  # Indigo
        '#7CB342',  # Light Green
        '#C0CA33',  # Lime
    ],
}


# =============================================================================
# DATA LOADING
# =============================================================================

class DataLoader:
    """Loads bounding boxes and camera parameters from a results directory."""

    def __init__(self, results_dir: str, images_dir: str):
        self.results_dir = Path(results_dir)
        self.images_dir = Path(images_dir)
        self.bboxes = self._load_bboxes()
        self.frame_names = sorted(
            self.bboxes.keys(),
            key=lambda x: int(re.findall(r'\d+', x)[0])
        )

    def _load_bboxes(self) -> Dict[str, list]:
        """Load all bounding boxes grouped by frame."""
        bboxes = {}
        bbox_dir = self.results_dir / "bounding_boxes"
        if not bbox_dir.exists():
            raise FileNotFoundError(f"Bounding box directory not found: {bbox_dir}")
        for f in sorted(bbox_dir.glob("*.json")):
            frame_name = f.stem
            with open(f) as fh:
                bboxes[frame_name] = json.load(fh)
        return bboxes

    def get_track_ids(self) -> List[int]:
        """Get all unique track IDs across all frames."""
        track_ids = set()
        for frame_data in self.bboxes.values():
            for det in frame_data:
                track_ids.add(det['track_id'])
        return sorted(track_ids)

    def get_track_frames(self, track_id: int) -> List[str]:
        """Get frame names where a given track is present."""
        return [f for f in self.frame_names
                if self.get_bbox_for_track(f, track_id) is not None]

    def get_bbox_for_track(self, frame_name: str, track_id: int) -> Optional[dict]:
        """Get bounding box for a specific track in a specific frame."""
        for det in self.bboxes.get(frame_name, []):
            if det['track_id'] == track_id:
                return det
        return None

    def load_camera(self, frame_name: str) -> Optional[dict]:
        """Load camera intrinsics and pose for a frame."""
        frame_num = re.findall(r'\d+', str(frame_name))
        variants = [frame_name]
        if frame_num:
            variants.extend([frame_num[0], frame_num[0].zfill(4)])

        search_dirs = [
            self.results_dir / "camera",
            self.results_dir.parent / "camera",
        ]

        for d in search_dirs:
            for name in variants:
                p = d / f"{name}.npz"
                if p.exists():
                    try:
                        data = np.load(p)
                        return {
                            'K': data['intrinsics'],
                            'R': data['pose'][:3, :3],
                            't': data['pose'][:3, 3],
                        }
                    except Exception:
                        pass
        return None


# =============================================================================
# MASK CROP EXTRACTOR
# =============================================================================

class MaskCropExtractor:
    """Extracts masked animal crops from images using grounded-SAM masks."""

    def __init__(self, images_dir: Path, mask_dir: Path = None,
                 results_dir: Path = None):
        self.images_dir = Path(images_dir) if images_dir else None
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.results_dir = Path(results_dir) if results_dir else None
        self.mask_track_mapping = self._load_mapping()
        if self.mask_dir is None:
            self._auto_discover_masks()

    def _auto_discover_masks(self):
        """Auto-discover grounded-sam mask directory."""
        candidates = []
        if self.images_dir:
            candidates.extend([
                self.images_dir / "grounded-sam",
                self.images_dir.parent / "grounded-sam",
            ])
        if self.results_dir:
            candidates.extend([
                self.results_dir / "grounded-sam",
                self.results_dir.parent / "grounded-sam",
            ])
        for c in candidates:
            if c.exists() and list(c.glob("*_results.json")):
                self.mask_dir = c
                print(f"  Auto-discovered mask directory: {c}")
                break

    def _load_mapping(self) -> dict:
        """Load mask-track mapping JSON."""
        search = []
        if self.results_dir:
            search.append(self.results_dir / "mask_track_mapping.json")
            search.append(self.results_dir.parent / "mask_track_mapping.json")
        for p in search:
            if p.exists():
                with open(p) as f:
                    print(f"  Loaded mask-track mapping: {p}")
                    return json.load(f)
        return {}

    def _load_image(self, frame_name: str) -> Optional[np.ndarray]:
        """Load an image with flexible naming."""
        if self.images_dir is None:
            return None
        frame_num = re.findall(r'\d+', str(frame_name))
        frame_key = frame_num[0] if frame_num else frame_name
        candidates = [frame_name, frame_key]
        if frame_key.isdigit():
            for w in [4, 5, 6, 8]:
                padded = frame_key.zfill(w)
                if padded not in candidates:
                    candidates.append(padded)
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            for name in candidates:
                p = self.images_dir / f"{name}{ext}"
                if p.exists():
                    return cv2.imread(str(p))
        return None

    def _load_mask(self, frame_name: str, track_id: int) -> Optional[np.ndarray]:
        """Load segmentation mask for a track from grounded-SAM results."""
        if not self.mask_dir or not PYCOCOTOOLS_AVAILABLE:
            return None
        frame_num = re.findall(r'\d+', str(frame_name))
        frame_key = frame_num[0] if frame_num else frame_name

        # Look up mask index from mapping
        mask_idx = self.mask_track_mapping.get(str(frame_key), {}).get(str(track_id))
        if mask_idx is None and frame_key.isdigit():
            for w in [4, 5, 6, 8]:
                padded = frame_key.zfill(w)
                mask_idx = self.mask_track_mapping.get(padded, {}).get(str(track_id))
                if mask_idx is not None:
                    break

        # Find the results JSON file
        name_candidates = [frame_key, frame_name]
        if frame_key.isdigit():
            for w in [4, 5, 6, 8]:
                padded = frame_key.zfill(w)
                if padded not in name_candidates:
                    name_candidates.append(padded)
        json_file = None
        for name in name_candidates:
            p = self.mask_dir / f"{name}_results.json"
            if p.exists():
                json_file = p
                break
        if json_file is None:
            return None

        try:
            with open(json_file) as f:
                results = json.load(f)
            annotations = results.get('annotations', [])
            if mask_idx is not None and mask_idx < len(annotations):
                rle = annotations[mask_idx].get('segmentation')
                if rle:
                    return mask_utils.decode(rle)
            # Fallback: single annotation
            if len(annotations) == 1:
                rle = annotations[0].get('segmentation')
                if rle:
                    return mask_utils.decode(rle)
        except Exception:
            pass
        return None

    def extract_crop(self, frame_name: str, track_id: int,
                     padding: int = None, background: str = None) -> Optional[np.ndarray]:
        """Extract a masked crop of the animal from an image."""
        if padding is None:
            padding = CONFIG['CROP_PADDING']
        if background is None:
            background = CONFIG['CROP_BG_COLOR']
        try:
            image = self._load_image(frame_name)
            if image is None:
                return self._error_img()

            mask = self._load_mask(frame_name, track_id)
            if mask is None:
                # Fallback: center crop
                h, w = image.shape[:2]
                s = min(h, w) // 2
                cx, cy = w // 2, h // 2
                return image[max(0, cy - s):min(h, cy + s),
                             max(0, cx - s):min(w, cx + s)].copy()

            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                return self._error_img()

            mx1, mx2 = xs.min(), xs.max()
            my1, my2 = ys.min(), ys.max()
            mcx, mcy = (mx1 + mx2) // 2, (my1 + my2) // 2
            max_dim = max(mx2 - mx1, my2 - my1) + 2 * padding
            half = max_dim // 2

            h, w = image.shape[:2]
            x1, x2 = max(0, mcx - half), min(w, mcx + half)
            y1, y2 = max(0, mcy - half), min(h, mcy + half)

            crop = image[y1:y2, x1:x2].copy()
            crop_mask = mask[y1:y2, x1:x2]

            if background == 'white':
                result = np.ones_like(crop) * 255
                result[crop_mask > 0] = crop[crop_mask > 0]
                return result
            return crop
        except Exception:
            return self._error_img()

    @staticmethod
    def _error_img(size: int = 200) -> np.ndarray:
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:, :, 2] = 80
        cv2.putText(img, "ERR", (60, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        return img


# =============================================================================
# VIEWING ANGLE COMPUTATION
# =============================================================================

class ViewingAngleComputer:
    """Computes camera-to-object viewing angles in the object's local frame.

    The viewing direction is the unit vector from camera position to bbox center,
    expressed in the bbox's local coordinate system (using the bbox rotation).
    This makes the viewing angle invariant to the object's world-frame position
    and orientation — two cameras seeing the same relative side of the animal
    will have similar local viewing directions.
    """

    def __init__(self, data_loader: DataLoader):
        self.data = data_loader

    def compute_viewing_direction(self, frame_name: str,
                                  track_id: int) -> Optional[np.ndarray]:
        """Compute normalised viewing direction in bbox-local frame."""
        bbox = self.data.get_bbox_for_track(frame_name, track_id)
        cam = self.data.load_camera(frame_name)
        if bbox is None or cam is None:
            return None

        bbox_center = np.array(bbox['center'])
        R_bbox = np.array(bbox['rotation_matrix'])  # local → world
        cam_pos = cam['t']  # camera position in world (camera-to-world convention)

        view_dir_world = bbox_center - cam_pos
        dist = np.linalg.norm(view_dir_world)
        if dist < 1e-6:
            return None
        view_dir_world /= dist

        # Express in bbox local frame
        view_dir_local = R_bbox.T @ view_dir_world
        view_dir_local /= (np.linalg.norm(view_dir_local) + 1e-12)
        return view_dir_local

    def compute_azimuth_elevation(self, frame_name: str,
                                  track_id: int) -> Optional[Tuple[float, float]]:
        """Return (azimuth, elevation) in degrees, or None."""
        v = self.compute_viewing_direction(frame_name, track_id)
        if v is None:
            return None
        azimuth = np.degrees(np.arctan2(v[1], v[0]))
        elevation = np.degrees(np.arcsin(np.clip(v[2], -1, 1)))
        return azimuth, elevation

    @staticmethod
    def angular_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """Angular distance in degrees between two unit vectors."""
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))


# =============================================================================
# CONTACT SHEET GENERATOR
# =============================================================================

def generate_contact_sheet(crops: Dict[str, np.ndarray],
                           frame_names: List[str],
                           track_id: int,
                           output_path: Path,
                           title: str = "",
                           label_map: Dict[str, str] = None,
                           label_colors: Dict[str, str] = None):
    """Generate a numbered contact sheet image showing all crops.

    Args:
        crops: {frame_name: BGR crop image}
        frame_names: ordered list of frame names to show
        track_id: track identifier
        output_path: where to save the contact sheet
        title: optional title text
        label_map: optional {frame_name: label} for color-coding
        label_colors: optional {label: hex_color}
    """
    cols = CONFIG['GALLERY_COLS']
    n = len(frame_names)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.9))
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]

    if title:
        fig.suptitle(title, fontsize=CONFIG['TYPOGRAPHY']['subtitle'],
                     fontweight='bold', y=0.99)

    for i, frame_name in enumerate(frame_names):
        r, c = i // cols, i % cols
        ax = axes[r][c]

        crop = crops.get(frame_name)
        if crop is not None:
            if len(crop.shape) == 3 and crop.shape[2] == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            ax.imshow(crop)
        else:
            ax.set_facecolor('#F0F0F0')

        # Index and frame number label
        frame_num = re.findall(r'\d+', str(frame_name))
        num_str = frame_num[0] if frame_num else frame_name
        ax.set_title(f"[{i}] {num_str}", fontsize=7, pad=2)
        ax.axis('off')

        # Color border if labelled
        if label_map and frame_name in label_map:
            label = label_map[frame_name]
            color = '#999999'
            if label_colors and label in label_colors:
                color = label_colors[label]
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            ax.set_title(f"[{i}] {num_str}\n{label}", fontsize=6,
                         fontweight='bold', pad=2)

    # Hide empty axes
    for i in range(n, rows * cols):
        r, c = i // cols, i % cols
        axes[r][c].axis('off')

    # Legend
    if label_map and label_colors:
        labels_present = sorted(set(label_map.values()))
        legend_elements = [
            mpatches.Patch(facecolor=label_colors.get(l, '#999'), alpha=0.7,
                           label=l.upper())
            for l in labels_present
        ]
        fig.legend(handles=legend_elements, loc='lower center',
                   ncol=min(len(labels_present), 8),
                   fontsize=CONFIG['TYPOGRAPHY']['body'], frameon=True,
                   bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Contact sheet saved: {output_path}")
    return output_path


# =============================================================================
# INTERACTIVE EXEMPLAR SELECTOR (matplotlib GUI)
# =============================================================================

class InteractiveExemplarSelector:
    """Clickable matplotlib gallery for selecting exemplar frames.

    Click a frame to select it, then type a label in the terminal.
    Press Enter/Q to finish selection.
    """

    def __init__(self, crops: Dict[str, np.ndarray],
                 frame_names: List[str], track_id: int):
        self.crops = crops
        self.frame_names = frame_names
        self.track_id = track_id
        self.exemplars: Dict[str, str] = {}  # frame_name → label
        self.axes: Dict[str, plt.Axes] = {}
        self.fig = None
        self.done = False

    def run(self) -> Dict[str, str]:
        """Run interactive selection. Returns {frame_name: label}."""
        n = len(self.frame_names)
        cols = CONFIG['GALLERY_COLS']
        rows = (n + cols - 1) // cols

        fig, axes_grid = plt.subplots(rows, cols,
                                       figsize=(cols * 1.6, rows * 1.9))
        self.fig = fig
        fig.suptitle(
            f'Track {self.track_id} — Click to select exemplars | '
            f'Enter = done',
            fontsize=11, fontweight='bold', y=0.99
        )

        if rows == 1:
            axes_grid = [axes_grid]
        if cols == 1:
            axes_grid = [[ax] for ax in axes_grid]

        for i, frame_name in enumerate(self.frame_names):
            r, c = i // cols, i % cols
            ax = axes_grid[r][c]
            crop = self.crops.get(frame_name)
            if crop is not None:
                disp = crop.copy()
                if len(disp.shape) == 3 and disp.shape[2] == 3:
                    disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                ax.imshow(disp)
            frame_num = re.findall(r'\d+', str(frame_name))
            num_str = frame_num[0] if frame_num else frame_name
            ax.set_title(f"[{i}] {num_str}", fontsize=7, pad=2)
            ax.axis('off')
            self.axes[frame_name] = ax

        for i in range(n, rows * cols):
            r, c = i // cols, i % cols
            axes_grid[r][c].axis('off')

        fig.canvas.mpl_connect('button_press_event', self._on_click)
        fig.canvas.mpl_connect('key_press_event', self._on_key)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        return self.exemplars

    def _on_click(self, event):
        if event.inaxes is None:
            return
        for frame_name, ax in self.axes.items():
            if event.inaxes == ax:
                if frame_name in self.exemplars:
                    # Deselect
                    del self.exemplars[frame_name]
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    frame_num = re.findall(r'\d+', str(frame_name))
                    idx = self.frame_names.index(frame_name)
                    num_str = frame_num[0] if frame_num else frame_name
                    ax.set_title(f"[{idx}] {num_str}", fontsize=7,
                                 fontweight='normal', pad=2)
                    print(f"  Deselected frame {frame_name}")
                else:
                    # Select — prompt for label
                    label = input(
                        f"\n  Enter viewpoint label for frame {frame_name}: "
                    ).strip()
                    if label:
                        self.exemplars[frame_name] = label
                        color = self._get_color(label)
                        for spine in ax.spines.values():
                            spine.set_visible(True)
                            spine.set_edgecolor(color)
                            spine.set_linewidth(3)
                        idx = self.frame_names.index(frame_name)
                        frame_num = re.findall(r'\d+', str(frame_name))
                        num_str = frame_num[0] if frame_num else frame_name
                        ax.set_title(f"[{idx}] {num_str}\n{label}",
                                     fontsize=6, fontweight='bold', pad=2)
                        print(f"  Labelled frame {frame_name} → '{label}'")
                break
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in ('enter', 'q'):
            self.done = True
            plt.close(self.fig)

    def _get_color(self, label: str) -> str:
        labels = sorted(set(self.exemplars.values()))
        idx = labels.index(label) if label in labels else 0
        return CONFIG['VIEWPOINT_COLORS'][idx % len(CONFIG['VIEWPOINT_COLORS'])]


# =============================================================================
# TERMINAL-BASED EXEMPLAR SELECTOR
# =============================================================================

def terminal_exemplar_selection(frame_names: List[str],
                                contact_sheet_path: Path) -> Dict[str, str]:
    """Prompt user to select exemplars via terminal input.

    Shows the contact sheet path so the user can view it, then asks for
    index–label pairs.
    """
    print(f"\n{'='*60}")
    print("EXEMPLAR SELECTION (terminal mode)")
    print(f"{'='*60}")
    print(f"  Contact sheet: {contact_sheet_path}")
    print(f"  Total frames: {len(frame_names)}")
    print()
    print("  Enter exemplar selections as:  INDEX LABEL")
    print("  Examples:")
    print("    0 front")
    print("    15 left-side")
    print("    42 rear")
    print()
    print("  Type 'done' or press Enter on empty line to finish.")
    print("  Type 'list' to show frame index → name mapping.")
    print()

    exemplars = {}
    while True:
        try:
            line = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line or line.lower() == 'done':
            break

        if line.lower() == 'list':
            for i, f in enumerate(frame_names):
                tag = f" [{exemplars[f]}]" if f in exemplars else ""
                print(f"    [{i:3d}] {f}{tag}")
            continue

        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            print("    Error: expected INDEX LABEL (e.g. '0 front')")
            continue

        try:
            idx = int(parts[0])
        except ValueError:
            print(f"    Error: '{parts[0]}' is not a valid index")
            continue

        if idx < 0 or idx >= len(frame_names):
            print(f"    Error: index {idx} out of range [0, {len(frame_names)-1}]")
            continue

        label = parts[1].strip()
        frame_name = frame_names[idx]
        exemplars[frame_name] = label
        print(f"    Added exemplar: [{idx}] {frame_name} → '{label}'")

    print(f"\n  Selected {len(exemplars)} exemplars.")
    return exemplars


# =============================================================================
# FRAME CLASSIFIER
# =============================================================================

class FrameClassifier:
    """Classifies frames by viewing-angle similarity to selected exemplars."""

    def __init__(self, viewing_angle_computer: ViewingAngleComputer,
                 data_loader: DataLoader, track_id: int):
        self.vac = viewing_angle_computer
        self.data = data_loader
        self.track_id = track_id

    def classify(self, exemplars: Dict[str, str],
                 max_angle: float = None) -> Tuple[Dict[str, List[str]], Dict]:
        """Classify all frames by nearest-exemplar viewing angle.

        Args:
            exemplars: {frame_name: viewpoint_label}
            max_angle: maximum angular distance (degrees) to accept

        Returns:
            classified: {label: [frame_names]}
            details: {frame_name: {label, distance, view_dir, is_exemplar}}
        """
        if max_angle is None:
            max_angle = CONFIG['MAX_ANGLE_THRESHOLD']

        # Compute viewing directions for exemplars
        exemplar_views = {}
        for frame_name, label in exemplars.items():
            v = self.vac.compute_viewing_direction(frame_name, self.track_id)
            if v is not None:
                exemplar_views[frame_name] = {'label': label, 'view_dir': v}
            else:
                print(f"  Warning: could not compute viewing angle for "
                      f"exemplar {frame_name}")

        if not exemplar_views:
            print("  Error: no valid viewing directions for any exemplar!")
            return {}, {}

        # Classify every frame
        classified = defaultdict(list)
        details = {}
        all_frames = self.data.get_track_frames(self.track_id)

        for frame_name in all_frames:
            v = self.vac.compute_viewing_direction(frame_name, self.track_id)
            if v is None:
                continue

            best_dist = float('inf')
            best_label = None
            for ex_frame, ex_data in exemplar_views.items():
                dist = ViewingAngleComputer.angular_distance(v, ex_data['view_dir'])
                if dist < best_dist:
                    best_dist = dist
                    best_label = ex_data['label']

            if best_dist <= max_angle:
                classified[best_label].append(frame_name)
                details[frame_name] = {
                    'label': best_label,
                    'distance': best_dist,
                    'view_dir': v.tolist(),
                    'is_exemplar': frame_name in exemplars,
                }

        # Sort each group chronologically
        for label in classified:
            classified[label].sort(
                key=lambda x: int(re.findall(r'\d+', x)[0])
            )

        return dict(classified), details


# =============================================================================
# FILMSTRIP PDF GENERATOR
# =============================================================================

class FilmstripGenerator:
    """Generates publication-style filmstrip PDF output."""

    def __init__(self, crop_extractor: MaskCropExtractor, track_id: int):
        self.crop = crop_extractor
        self.track_id = track_id
        self.typo = CONFIG['TYPOGRAPHY']

    def _build_label_colors(self, labels: List[str]) -> Dict[str, str]:
        """Assign a colour to each viewpoint label."""
        colors = CONFIG['VIEWPOINT_COLORS']
        return {l: colors[i % len(colors)] for i, l in enumerate(sorted(labels))}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(self, classified: Dict[str, List[str]],
                 details: Dict, exemplars: Dict[str, str],
                 output_path, video_name: str = ""):
        """Generate the complete filmstrip PDF."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        labels = sorted(classified.keys())
        label_colors = self._build_label_colors(labels)

        with PdfPages(str(output_path)) as pdf:
            self._summary_page(pdf, classified, exemplars, label_colors,
                               video_name)
            self._filmstrip_pages(pdf, classified, details, exemplars,
                                  label_colors, video_name)

        print(f"\n  Filmstrip PDF saved: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Page 1 — Summary with exemplar crops
    # ------------------------------------------------------------------

    def _summary_page(self, pdf, classified, exemplars, label_colors,
                      video_name):
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle(f'EXEMPLAR VIEWPOINT ANALYSIS — Track {self.track_id}',
                     fontsize=self.typo['title'], fontweight='bold', y=0.97)
        if video_name:
            fig.text(0.5, 0.94, video_name, ha='center',
                     fontsize=self.typo['subtitle'], color='gray')

        labels = sorted(classified.keys())
        n = len(labels)
        if n == 0:
            fig.text(0.5, 0.5, "No frames classified", ha='center',
                     va='center', fontsize=self.typo['heading'])
            pdf.savefig(fig)
            plt.close(fig)
            return

        cols = min(n, 5)
        rows = (n + cols - 1) // cols
        gs = GridSpec(rows * 2, cols, figure=fig,
                      left=0.05, right=0.95, top=0.88, bottom=0.20,
                      hspace=0.5, wspace=0.15)

        for i, label in enumerate(labels):
            r, c = (i // cols) * 2, i % cols
            color = label_colors[label]
            n_frames = len(classified[label])
            n_ex = sum(1 for f in classified[label] if f in exemplars)

            # Header cell
            ax_h = fig.add_subplot(gs[r, c])
            ax_h.axis('off')
            ax_h.text(0.5, 0.5,
                      f'{label.upper()}\n{n_frames} frames ({n_ex} exemplar)',
                      ha='center', va='center', fontsize=self.typo['body'],
                      fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                                alpha=0.3))

            # Exemplar crop
            ax_c = fig.add_subplot(gs[r + 1, c])
            ex_frames = [f for f in classified[label] if f in exemplars]
            if ex_frames:
                crop = self.crop.extract_crop(ex_frames[0], self.track_id)
                if crop is not None:
                    if len(crop.shape) == 3 and crop.shape[2] == 3:
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    ax_c.imshow(crop)
                    for spine in ax_c.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor(color)
                        spine.set_linewidth(3)
            ax_c.axis('off')

        # Stats
        total = sum(len(v) for v in classified.values())
        fig.text(0.5, 0.14,
                 f"Total classified: {total} frames  |  "
                 f"Viewpoints: {n}  |  Exemplars: {len(exemplars)}",
                 ha='center', fontsize=self.typo['body'])

        # Legend
        legend = [mpatches.Patch(facecolor=label_colors[l], alpha=0.6,
                                  label=l.upper()) for l in labels]
        fig.legend(handles=legend, loc='lower center',
                   ncol=min(n, 8), fontsize=self.typo['body'], frameon=True,
                   bbox_to_anchor=(0.5, 0.04))

        pdf.savefig(fig)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Pages 2+ — Chronological filmstrip
    # ------------------------------------------------------------------

    def _filmstrip_pages(self, pdf, classified, details, exemplars,
                         label_colors, video_name):
        # Merge all classified frames into one chronological list
        all_frames = []
        for label, frames in classified.items():
            for f in frames:
                all_frames.append((f, label, f in exemplars))
        all_frames.sort(key=lambda x: int(re.findall(r'\d+', x[0])[0]))

        max_per_row = CONFIG['MAX_FRAMES_PER_ROW']
        max_rows = 3
        max_per_page = max_per_row * max_rows
        total_pages = max((len(all_frames) + max_per_page - 1) // max_per_page, 1)

        for page_idx in range(total_pages):
            start = page_idx * max_per_page
            page_frames = all_frames[start:start + max_per_page]
            self._draw_filmstrip_page(pdf, page_frames, label_colors,
                                      page_idx + 1, total_pages, video_name)

    def _draw_filmstrip_page(self, pdf, frames, label_colors,
                             page_num, total_pages, video_name):
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle(
            f'TRACK {self.track_id} — FILMSTRIP '
            f'(Page {page_num}/{total_pages})',
            fontsize=self.typo['subtitle'], fontweight='bold', y=0.97)
        if video_name:
            fig.text(0.5, 0.94, video_name, ha='center',
                     fontsize=self.typo['body'], color='gray')

        if not frames:
            fig.text(0.5, 0.5, 'No frames to display', ha='center',
                     va='center', fontsize=self.typo['heading'], color='gray',
                     style='italic')
            pdf.savefig(fig)
            plt.close(fig)
            return

        max_per_row = CONFIG['MAX_FRAMES_PER_ROW']
        n_frames = len(frames)
        n_rows = (n_frames + max_per_row - 1) // max_per_row

        strip_top = 0.90
        strip_bottom = 0.10
        row_height = (strip_top - strip_bottom) / max(n_rows, 1)

        for i, (frame_name, label, is_exemplar) in enumerate(frames):
            row = i // max_per_row
            col = i % max_per_row
            frames_in_row = min(n_frames - row * max_per_row, max_per_row)

            fw = 0.88 / frames_in_row
            x = 0.05 + col * fw
            y = strip_top - (row + 1) * row_height + 0.03

            ax = fig.add_axes([x, y, fw * 0.9, row_height * 0.72])

            crop = self.crop.extract_crop(frame_name, self.track_id)
            if crop is not None:
                if len(crop.shape) == 3 and crop.shape[2] == 3:
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                ax.imshow(crop)
            else:
                ax.set_facecolor('#F5F5F5')
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        fontsize=self.typo['caption'], color='gray')
            ax.axis('off')

            # Border
            color = label_colors.get(label, '#999999')
            bw = 4 if is_exemplar else 2
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(color)
                spine.set_linewidth(bw)

            # Star on exemplar
            if is_exemplar:
                ax.text(0.05, 0.95, '*', ha='left', va='top', fontsize=14,
                        color=color, fontweight='bold',
                        transform=ax.transAxes)

            # Frame number
            fnum = re.findall(r'\d+', str(frame_name))
            ax.text(0.5, 1.10, fnum[0] if fnum else frame_name,
                    ha='center', va='bottom', fontsize=self.typo['micro'],
                    transform=ax.transAxes)

            # Label badge
            badge = label.upper()[:6]
            ax.text(0.5, -0.14, badge, ha='center', va='top',
                    fontsize=self.typo['micro'], fontweight='bold',
                    transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor=color,
                              alpha=0.6, edgecolor='none'))

        # Legend
        labels_used = sorted(set(l for _, l, _ in frames))
        legend = [mpatches.Patch(facecolor=label_colors.get(l, '#999'),
                                  alpha=0.6, label=l.upper())
                  for l in labels_used]
        fig.legend(handles=legend, loc='lower center',
                   ncol=min(len(labels_used), 8),
                   fontsize=self.typo['body'], frameon=True,
                   bbox_to_anchor=(0.5, 0.02))

        pdf.savefig(fig)
        plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Exemplar-based Viewpoint Analyzer — '
                    'filmstrip output without semantic labels')

    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to results directory '
                             '(with bounding_boxes/, camera/)')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Path to images directory')
    parser.add_argument('--track_id', type=int, default=None,
                        help='Track ID to analyse (default: all)')
    parser.add_argument('--max_angle', type=float, default=30.0,
                        help='Max angular distance for classification '
                             '(degrees, default 30)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PDF path (default: auto)')
    parser.add_argument('--interactive', action='store_true',
                        help='Use interactive matplotlib gallery '
                             '(default: terminal mode)')
    parser.add_argument('--load_saved', action='store_true',
                        help='Load previously saved exemplar selections')
    parser.add_argument('--no_review', action='store_true',
                        help='Skip the classified review contact sheet')

    args = parser.parse_args()

    # ---- Initialise ----
    print(f"\nExemplar Viewpoint Analyzer")
    print(f"{'='*60}")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Images dir  : {args.images_dir}")
    print(f"  Max angle   : {args.max_angle} deg")

    data = DataLoader(args.results_dir, args.images_dir)
    crop_ext = MaskCropExtractor(args.images_dir, results_dir=args.results_dir)
    vac = ViewingAngleComputer(data)

    track_ids = ([args.track_id] if args.track_id is not None
                 else data.get_track_ids())
    print(f"  Tracks      : {track_ids}")

    analysis_dir = Path(args.results_dir) / "viewpoint_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # ---- Process each track ----
    for track_id in track_ids:
        print(f"\n{'='*60}")
        print(f"  TRACK {track_id}")
        print(f"{'='*60}")

        track_frames = data.get_track_frames(track_id)
        print(f"  Frames with track: {len(track_frames)}")

        # Pre-load crops
        print("  Loading crops...")
        crops = {}
        for f in track_frames:
            crops[f] = crop_ext.extract_crop(f, track_id)

        save_path = analysis_dir / f"exemplars_track{track_id}.json"

        # ---- Exemplar selection ----
        if args.load_saved and save_path.exists():
            with open(save_path) as f:
                exemplars = json.load(f)
            print(f"  Loaded {len(exemplars)} saved exemplars from {save_path}")
        else:
            # Generate contact sheet first
            sheet_path = analysis_dir / f"contact_sheet_track{track_id}.png"
            generate_contact_sheet(
                crops, track_frames, track_id, sheet_path,
                title=f"Track {track_id} — Select exemplars "
                      f"(index + label)"
            )

            if args.interactive:
                selector = InteractiveExemplarSelector(
                    crops, track_frames, track_id)
                exemplars = selector.run()
            else:
                exemplars = terminal_exemplar_selection(
                    track_frames, sheet_path)

            if not exemplars:
                print("  No exemplars selected — skipping track.")
                continue

            # Save
            with open(save_path, 'w') as f:
                json.dump(exemplars, f, indent=2)
            print(f"  Saved exemplars to {save_path}")

        # ---- Classify ----
        classifier = FrameClassifier(vac, data, track_id)
        classified, details = classifier.classify(exemplars, args.max_angle)

        print(f"\n  Classification results (max_angle={args.max_angle} deg):")
        for label in sorted(classified.keys()):
            frames = classified[label]
            n_ex = sum(1 for f in frames if f in exemplars)
            print(f"    {label:15s}: {len(frames):3d} frames "
                  f"({n_ex} exemplar)")
        unclassified = len(track_frames) - sum(len(v) for v in classified.values())
        if unclassified > 0:
            print(f"    {'(unclassified)':15s}: {unclassified:3d} frames")

        # ---- Review contact sheet ----
        if not args.no_review:
            label_map = {f: d['label'] for f, d in details.items()}
            labels = sorted(set(label_map.values()))
            lc = {l: CONFIG['VIEWPOINT_COLORS'][i % len(CONFIG['VIEWPOINT_COLORS'])]
                  for i, l in enumerate(labels)}
            review_path = analysis_dir / f"classified_track{track_id}.png"
            generate_contact_sheet(
                crops, track_frames, track_id, review_path,
                title=f"Track {track_id} — Classified "
                      f"(max_angle={args.max_angle} deg)",
                label_map=label_map, label_colors=lc
            )

        # ---- Generate filmstrip ----
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = analysis_dir / f"filmstrip_exemplar_track{track_id}.pdf"

        video_name = Path(args.results_dir).name
        gen = FilmstripGenerator(crop_ext, track_id)
        gen.generate(classified, details, exemplars, out_path, video_name)

    print(f"\nDone. Output in: {analysis_dir}")


if __name__ == '__main__':
    main()
