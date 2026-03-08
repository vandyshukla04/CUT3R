"""
Data loaders for wildlife tracking evaluation.

Loads results produced by CUT3R's tracking pipeline:
  - bounding_boxes/*.json   (3D bbox per frame)
  - instance_labels/*.npy   (instance segmentation masks)
  - mask_track_mapping.json (frame -> track -> mask index)
  - tracking_summary.json   (per-track statistics)
"""

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class BBox3D:
    """A single 3D bounding box detection."""
    center: np.ndarray
    dimensions: np.ndarray
    rotation_matrix: np.ndarray
    class_name: str
    track_id: int
    frame_idx: int
    confidence: float = 1.0
    instance_ids: List[int] = field(default_factory=list)


class SequenceLoader:
    """Loads a CUT3R tracking-result sequence from disk.

    After calling the ``load_*`` methods the following attributes are populated:

    Attributes
    ----------
    results_dir : Path
        Root of the results directory.
    frame_names : List[str]
        Sorted frame name strings (e.g. ``["2240", "2245", ...]``).
    frame_indices : List[int]
        ``range(len(frame_names))`` – sequential 0-based indices.
    bboxes : Dict[int, List[BBox3D]]
        ``frame_index -> list of BBox3D`` for every frame.
    instance_masks : Dict[int, np.ndarray]
        ``frame_index -> (H, W) ndarray`` of instance labels.
    mask_track_mapping : Dict[str, Dict[str, int]]
        ``frame_name -> {track_id_str: mask_annotation_index}``.
    tracking_summary : dict
        Raw content of ``tracking_summary.json``.
    """

    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.frame_names: List[str] = []
        self.frame_indices: List[int] = []
        self.bboxes: Dict[int, List[BBox3D]] = {}
        self.instance_masks: Dict[int, np.ndarray] = {}
        self.mask_track_mapping: Dict[str, Dict[str, int]] = {}
        self.tracking_summary: dict = {}

    # ------------------------------------------------------------------
    # Public loading API
    # ------------------------------------------------------------------

    def load_bounding_boxes(self):
        """Load ``bounding_boxes/*.json`` into *self.bboxes*.

        Also discovers *frame_names* / *frame_indices* from the filenames.
        """
        bbox_dir = self.results_dir / "bounding_boxes"
        if not bbox_dir.exists():
            print(f"[SequenceLoader] bounding_boxes dir not found: {bbox_dir}")
            return

        files = sorted(bbox_dir.glob("*.json"), key=lambda p: int(p.stem))
        # Discover frame names from bbox files (authoritative ordering)
        discovered_names = [f.stem for f in files]
        if not self.frame_names:
            self.frame_names = discovered_names
            self.frame_indices = list(range(len(self.frame_names)))

        for idx, fp in enumerate(files):
            frame_name = fp.stem
            # idx in the sorted file list corresponds to the frame_index
            frame_idx = self.frame_names.index(frame_name) if frame_name in self.frame_names else idx
            with open(fp) as f:
                data = json.load(f)

            boxes: List[BBox3D] = []
            if isinstance(data, list):
                for entry in data:
                    boxes.append(BBox3D(
                        center=np.array(entry["center"], dtype=np.float64),
                        dimensions=np.array(entry["dimensions"], dtype=np.float64),
                        rotation_matrix=np.array(entry["rotation_matrix"], dtype=np.float64),
                        class_name=entry.get("class_name", "unknown"),
                        track_id=int(entry.get("track_id", -1)),
                        frame_idx=frame_idx,
                        confidence=float(entry.get("confidence", 1.0)),
                        instance_ids=entry.get("instance_ids", []),
                    ))
            self.bboxes[frame_idx] = boxes

        print(f"[SequenceLoader] Loaded bounding boxes for {len(self.bboxes)} frames")

    def load_instance_masks(self):
        """Load ``instance_labels/*.npy`` into *self.instance_masks*."""
        mask_dir = self.results_dir / "instance_labels"
        if not mask_dir.exists():
            print(f"[SequenceLoader] instance_labels dir not found: {mask_dir}")
            return

        files = sorted(mask_dir.glob("*.npy"), key=lambda p: int(p.stem))
        # Discover frame names from mask files if not yet set
        if not self.frame_names:
            self.frame_names = [f.stem for f in files]
            self.frame_indices = list(range(len(self.frame_names)))

        for fp in files:
            frame_name = fp.stem
            if frame_name in self.frame_names:
                frame_idx = self.frame_names.index(frame_name)
            else:
                continue
            self.instance_masks[frame_idx] = np.load(str(fp))

        print(f"[SequenceLoader] Loaded instance masks for {len(self.instance_masks)} frames")

    def load_mask_track_mapping(self):
        """Load ``mask_track_mapping.json`` into *self.mask_track_mapping*."""
        path = self.results_dir / "mask_track_mapping.json"
        if not path.exists():
            print(f"[SequenceLoader] mask_track_mapping.json not found: {path}")
            return
        with open(path) as f:
            self.mask_track_mapping = json.load(f)
        print(f"[SequenceLoader] Loaded mask-track mapping for {len(self.mask_track_mapping)} frames")

    def load_tracking_summary(self):
        """Load ``tracking_summary.json`` into *self.tracking_summary*.

        Also ensures *frame_names* / *frame_indices* are populated from the
        summary's track frame lists when bbox files were not available.
        """
        path = self.results_dir / "tracking_summary.json"
        if not path.exists():
            print(f"[SequenceLoader] tracking_summary.json not found: {path}")
            return
        with open(path) as f:
            self.tracking_summary = json.load(f)

        # If frame_names were not yet discovered, derive them from the
        # mask_track_mapping keys or instance_labels directory.
        if not self.frame_names:
            if self.mask_track_mapping:
                self.frame_names = sorted(self.mask_track_mapping.keys(), key=lambda x: int(x))
            else:
                mask_dir = self.results_dir / "instance_labels"
                if mask_dir.exists():
                    self.frame_names = sorted(
                        [f.stem for f in mask_dir.glob("*.npy")], key=lambda x: int(x)
                    )
            self.frame_indices = list(range(len(self.frame_names)))

        print(f"[SequenceLoader] Loaded tracking summary "
              f"({self.tracking_summary.get('total_tracks', '?')} tracks, "
              f"{self.tracking_summary.get('frames_processed', '?')} frames)")

    def get_image_shape(self) -> Optional[Tuple[int, int]]:
        """Return ``(height, width)`` determined from the first loaded mask.

        Returns *None* if no masks have been loaded.
        """
        for mask in self.instance_masks.values():
            return mask.shape[:2]
        return None
