#!/usr/bin/env python3
"""
Interactive Frame Viewer for Ecology Visualizations

Navigate through frames with arrow keys, save current view with 's'.

Controls:
    Left/Right arrows: Previous/Next frame
    Up/Down arrows: Jump 10 frames
    Home/End: First/Last frame
    S: Save current figure
    Q/Escape: Quit

Batch mode (no display):
    python -m tools.frame_viewer --result_dir ... --batch --frames 0,10,20,30
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Set up matplotlib backend BEFORE importing pyplot
import matplotlib

# Check if we have a display (including WSLg)
_display = os.environ.get('DISPLAY')
_wayland = os.environ.get('WAYLAND_DISPLAY')
_wslg = os.path.exists('/mnt/wslg/.X11-unix')

HAS_DISPLAY = bool(_display or _wayland or _wslg)

if HAS_DISPLAY:
    try:
        matplotlib.use('TkAgg')
    except Exception:
        try:
            matplotlib.use('Qt5Agg')
        except Exception:
            matplotlib.use('Agg')
            HAS_DISPLAY = False
else:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import from ecology_visualizer
from tools.ecology_visualizer import (
    EcologyVisualizer,
    TrackingRenderer,
    get_instance_color,
    get_track_color,
    load_frame_data,
    unproject_depth,
)


class InteractiveFrameViewer:
    """Interactive viewer for comparison figures with frame-by-frame navigation."""

    def __init__(self, result_dir: str, srt_file: Optional[str] = None,
                 output_dir: str = "saved_frames"):
        self.result_dir = Path(result_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize visualizer
        self.viz = EcologyVisualizer(result_dir, srt_file)
        self.viz.load_all_frames()
        self.viz.compute_canonical()
        self.viz.load_tracking()

        self.frame_ids = self.viz.frame_ids
        self.current_idx = 0
        self.fig = None
        self.axes = None

        # Precompute per-frame data
        self._frame_data = {}
        self._frame_points = {}
        self._load_frame_points()

    def _load_frame_points(self):
        """Load per-frame point clouds for individual frame rendering."""
        print("Pre-loading frame point data...")
        for i, fid in enumerate(self.frame_ids):
            try:
                data = load_frame_data(self.result_dir, fid)
                pts = unproject_depth(data['depth'], data['intrinsics'], data['pose'])
                pts_canonical = self.viz.aligner.transform(pts)

                self._frame_data[fid] = data
                self._frame_points[fid] = {
                    'points': pts_canonical,
                    'labels': data['instance_labels'].flatten(),
                    'conf': data['conf'].flatten(),
                    'image': data.get('image')
                }
            except Exception as e:
                print(f"Warning: Could not load frame {fid}: {e}")

            if (i + 1) % 20 == 0:
                print(f"  Loaded {i + 1}/{len(self.frame_ids)} frames")
        print(f"Loaded {len(self._frame_points)} frame point clouds")

    def _get_single_frame_data(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get points, labels, and confidence for a single frame only."""
        fid = self.frame_ids[frame_idx]
        if fid not in self._frame_points:
            return np.array([]), np.array([]), np.array([])

        fp = self._frame_points[fid]
        return fp['points'], fp['labels'], fp['conf']

    def _get_trajectories_up_to(self, up_to_idx: int) -> Dict[int, List]:
        """Get trajectories only up to current frame index."""
        trajectories = {}
        for tid, traj in self.viz._trajectories.items():
            # Filter to only include points up to current frame
            # trajectory points have 'frame' key (frame index)
            filtered = [p for p in traj if p.get('frame', 0) <= up_to_idx]
            if filtered:
                trajectories[tid] = filtered

        return trajectories

    def _render_point_cloud(self, ax, points: np.ndarray, labels: np.ndarray,
                            conf: np.ndarray, conf_threshold: float = 1.5,
                            point_size: float = 0.5, bg_alpha: float = 0.15,
                            inst_alpha: float = 0.8):
        """Render point cloud with instance colors."""
        valid = conf > conf_threshold
        pts = points[valid]
        lbls = labels[valid]

        if len(pts) == 0:
            return

        # Top-down view: X, Y
        xi, yi = 0, 1

        # Background
        bg = lbls == 0
        if np.any(bg):
            ax.scatter(pts[bg, xi], pts[bg, yi], c='lightgray',
                      s=point_size, alpha=bg_alpha, rasterized=True)

        # Instances
        for inst_id in np.unique(lbls):
            if inst_id == 0:
                continue
            mask = lbls == inst_id
            color = get_instance_color(inst_id)
            ax.scatter(pts[mask, xi], pts[mask, yi], c=[color],
                      s=point_size * 1.5, alpha=inst_alpha, rasterized=True)

    def _render_trajectories(self, ax, trajectories: Dict[int, List],
                             up_to_idx: int, linewidth: float = 2.0):
        """Render motion trails up to current frame."""
        for tid, traj in trajectories.items():
            if len(traj) < 2:
                continue

            color = get_track_color(tid)

            # Sort by frame index
            traj_sorted = sorted(traj, key=lambda x: x.get('frame', 0))

            # Get positions (canonical coords, top-down = X, Y)
            positions = []
            for p in traj_sorted:
                if 'center_canonical' in p:
                    positions.append(p['center_canonical'][:2])
                elif 'center' in p:
                    # Transform to canonical
                    c = np.array(p['center']).reshape(1, 3)
                    c_can = self.viz.aligner.transform(c)[0]
                    positions.append(c_can[:2])

            if len(positions) < 2:
                continue

            positions = np.array(positions)

            # Draw trail with temporal gradient
            n_points = len(positions)
            for i in range(n_points - 1):
                # Fade: older = lighter
                alpha = 0.3 + 0.7 * (i / max(n_points - 1, 1))
                lw = linewidth * (0.5 + 0.5 * (i / max(n_points - 1, 1)))

                ax.plot(positions[i:i+2, 0], positions[i:i+2, 1],
                       color=color, linewidth=lw, alpha=alpha, solid_capstyle='round')

            # Start marker (circle)
            ax.scatter(positions[0, 0], positions[0, 1], c=[color],
                      s=30, marker='o', edgecolors='white', linewidths=0.5, zorder=10)

            # End marker (triangle) with track ID
            ax.scatter(positions[-1, 0], positions[-1, 1], c=[color],
                      s=50, marker='^', edgecolors='white', linewidths=0.5, zorder=11)

            # Label
            class_name = self.viz._track_info.get(tid, {}).get('class_name', f'Track {tid}')
            ax.annotate(f"{class_name}", positions[-1], fontsize=7,
                       xytext=(5, 5), textcoords='offset points',
                       color=color, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                alpha=0.7, edgecolor=color, linewidth=0.5))

    def _draw_frame(self):
        """Draw the current frame's comparison figure."""
        if self.fig is None:
            return

        # Clear all axes
        for ax in self.axes:
            ax.clear()

        fid = self.frame_ids[self.current_idx]
        fp = self._frame_points.get(fid)

        if fp is None:
            self.fig.suptitle(f"Frame {fid} - Data not available", fontsize=12)
            self.fig.canvas.draw_idle()
            return

        # Get single frame data (not cumulative)
        pts, lbls, conf = self._get_single_frame_data(self.current_idx)
        # But trajectories accumulate up to current frame
        trajectories = self._get_trajectories_up_to(self.current_idx)

        # Panel A: Input image
        ax1 = self.axes[0]
        if fp['image'] is not None:
            ax1.imshow(fp['image'])
        ax1.set_title('(a) Input Frame', fontsize=11, fontweight='bold')
        ax1.axis('off')

        # Panel B: Instance segmentation (single frame only)
        ax2 = self.axes[1]
        if len(pts) > 0:
            self._render_point_cloud(ax2, pts, lbls, conf, point_size=0.5)
        ax2.set_title('(b) Instance Segmentation', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Forward (m)', fontsize=9)
        ax2.set_ylabel('Right (m)', fontsize=9)
        ax2.invert_yaxis()
        ax2.set_aspect('equal')
        ax2.tick_params(labelsize=8)

        # Panel C: Single frame point cloud + accumulated trajectories
        ax3 = self.axes[2]
        if len(pts) > 0:
            self._render_point_cloud(ax3, pts, lbls, conf, point_size=0.3,
                                    bg_alpha=0.1, inst_alpha=0.4)
        if trajectories:
            self._render_trajectories(ax3, trajectories, self.current_idx)
        ax3.set_title('(c) Motion Trajectories', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Forward (m)', fontsize=9)
        ax3.set_ylabel('Right (m)', fontsize=9)
        ax3.invert_yaxis()
        ax3.set_aspect('equal')
        ax3.tick_params(labelsize=8)

        # Update title
        self.fig.suptitle(f"Frame {fid} ({self.current_idx + 1}/{len(self.frame_ids)}) - "
                         f"Press S to save, Q to quit", fontsize=10)

        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key in ('right', 'd'):
            self.current_idx = min(self.current_idx + 1, len(self.frame_ids) - 1)
            self._draw_frame()
        elif event.key in ('left', 'a'):
            self.current_idx = max(self.current_idx - 1, 0)
            self._draw_frame()
        elif event.key in ('up', 'w'):
            self.current_idx = min(self.current_idx + 10, len(self.frame_ids) - 1)
            self._draw_frame()
        elif event.key in ('down', 'x'):
            self.current_idx = max(self.current_idx - 10, 0)
            self._draw_frame()
        elif event.key == 'home':
            self.current_idx = 0
            self._draw_frame()
        elif event.key == 'end':
            self.current_idx = len(self.frame_ids) - 1
            self._draw_frame()
        elif event.key == 's':
            self._save_current()
        elif event.key in ('q', 'escape'):
            plt.close(self.fig)

    def _save_current(self):
        """Save current figure."""
        fid = self.frame_ids[self.current_idx]
        output_path = self.output_dir / f"comparison_frame_{fid}.png"
        self.fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

    def run(self):
        """Start the interactive viewer."""
        if not HAS_DISPLAY:
            print("\n" + "="*60)
            print("ERROR: No display available!")
            print("="*60)
            print("Options:")
            print("  1. Set up X11 forwarding (export DISPLAY=:0)")
            print("  2. Use WSLg (Windows 11 with WSL2)")
            print("  3. Use batch mode to export specific frames:")
            print(f"     python -m tools.frame_viewer --result_dir {self.result_dir} --batch --frames 0,10,20")
            print("  4. Export all frames:")
            print(f"     python -m tools.frame_viewer --result_dir {self.result_dir} --batch --all")
            print("="*60 + "\n")
            return

        # Create figure
        self.fig = plt.figure(figsize=(14, 5))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.08)

        self.axes = [
            self.fig.add_subplot(gs[0]),
            self.fig.add_subplot(gs[1]),
            self.fig.add_subplot(gs[2])
        ]

        # Connect keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Draw initial frame
        self._draw_frame()

        print("\n" + "="*60)
        print("Interactive Frame Viewer")
        print("="*60)
        print("Controls:")
        print("  Left/Right (or A/D): Previous/Next frame")
        print("  Up/Down (or W/X): Jump 10 frames")
        print("  Home/End: First/Last frame")
        print("  S: Save current figure")
        print("  Q/Escape: Quit")
        print("="*60 + "\n")

        plt.show()

    def export_frame(self, frame_idx: int) -> str:
        """Export a single frame to file."""
        # Create figure
        self.fig = plt.figure(figsize=(14, 5))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.08)

        self.axes = [
            self.fig.add_subplot(gs[0]),
            self.fig.add_subplot(gs[1]),
            self.fig.add_subplot(gs[2])
        ]

        self.current_idx = frame_idx
        self._draw_frame()

        fid = self.frame_ids[frame_idx]
        output_path = self.output_dir / f"comparison_frame_{fid}.png"
        self.fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(self.fig)

        return str(output_path)

    def export_batch(self, frame_indices: List[int], show_progress: bool = True):
        """Export multiple frames."""
        print(f"\nExporting {len(frame_indices)} frames to {self.output_dir}/")
        saved = []

        for i, idx in enumerate(frame_indices):
            if idx < 0 or idx >= len(self.frame_ids):
                print(f"  Skipping invalid frame index: {idx}")
                continue

            path = self.export_frame(idx)
            saved.append(path)

            if show_progress and (i + 1) % 5 == 0:
                print(f"  Exported {i + 1}/{len(frame_indices)} frames")

        print(f"\nDone! Exported {len(saved)} frames to {self.output_dir}/")
        return saved


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Frame Viewer for Ecology Visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode:
    python -m tools.frame_viewer --result_dir results/zebras/

  Batch export specific frames:
    python -m tools.frame_viewer --result_dir results/zebras/ --batch --frames 0,10,20,30

  Export all frames:
    python -m tools.frame_viewer --result_dir results/zebras/ --batch --all

  Export every 5th frame:
    python -m tools.frame_viewer --result_dir results/zebras/ --batch --every 5
        """
    )
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Path to results directory')
    parser.add_argument('--srt_file', type=str, default=None,
                       help='Optional SRT file for gimbal alignment')
    parser.add_argument('--output_dir', type=str, default='saved_frames',
                       help='Directory to save frames')
    parser.add_argument('--start_frame', type=int, default=0,
                       help='Starting frame index (interactive mode)')

    # Batch mode options
    parser.add_argument('--batch', action='store_true',
                       help='Run in batch mode (no display needed)')
    parser.add_argument('--frames', type=str, default=None,
                       help='Comma-separated frame indices to export (e.g., 0,10,20,30)')
    parser.add_argument('--all', action='store_true',
                       help='Export all frames')
    parser.add_argument('--every', type=int, default=None,
                       help='Export every Nth frame')

    args = parser.parse_args()

    viewer = InteractiveFrameViewer(
        result_dir=args.result_dir,
        srt_file=args.srt_file,
        output_dir=args.output_dir
    )

    # Batch mode
    if args.batch:
        n_frames = len(viewer.frame_ids)

        if args.frames:
            # Parse comma-separated frame indices
            frame_indices = [int(x.strip()) for x in args.frames.split(',')]
        elif args.all:
            frame_indices = list(range(n_frames))
        elif args.every:
            frame_indices = list(range(0, n_frames, args.every))
        else:
            print("Batch mode requires --frames, --all, or --every option")
            print(f"Total frames available: {n_frames}")
            sys.exit(1)

        viewer.export_batch(frame_indices)
    else:
        # Interactive mode
        viewer.current_idx = args.start_frame
        viewer.run()


if __name__ == '__main__':
    main()
