# Analysis and Debugging Tools

This directory contains tools for analyzing data structures and debugging wildlife detection results.

## Available Tools

### explore_data.py

Comprehensive data structure analyzer for CUT3R outputs.

**Purpose:**
Explore and understand the structure of CUT3R output directories, including:
- Directory organization
- File formats and contents
- Camera parameters
- 3D bounding box annotations
- Depth maps
- 2D segmentation masks

**Usage:**
```bash
python tools/explore_data.py --data-dir <path_to_results>
```

**Features:**
- **Directory Analysis**: Lists all subdirectories and file counts
- **Camera Inspection**: Displays camera intrinsics and poses
- **3D Box Analysis**: Shows bounding box dimensions, rotations, and centers
- **Depth Map Statistics**: Analyzes depth map ranges and distributions
- **SAM Mask Inspection**: Examines Grounded-SAM segmentation results
- **Frame Matching**: Identifies frames with complete annotations

**Example:**
```bash
# Explore a single result directory
python tools/explore_data.py --data-dir results/tmp-rhin-105_1-revisit-1

# Explore multiple directories
python tools/explore_data.py --data-dir results --recursive
```

**Output:**
```
Directory Structure:
├── bounding_boxes/  (150 files)
├── camera/          (1 file)
├── depth/           (150 files)
└── point_cloud/     (150 files)

Camera Parameters:
- Intrinsics: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
- Image size: 768 x 432

3D Bounding Boxes (Frame 001):
- Class: rhino
- Center: [x, y, z]
- Dimensions: [width, height, length]
- Rotation: [[...]]
- Confidence: 0.95

Depth Map Statistics:
- Range: [min, max]
- Mean: value
- Shape: (H, W)
```

---

### analyze_rhino_data.py

Dataset-specific analyzer for rhino wildlife detection data.

**Purpose:**
Perform in-depth analysis of rhino detection datasets, including:
- Coordinate system verification
- Annotation quality assessment
- Dataset statistics and distributions
- Video-result matching verification

**Usage:**
```bash
python tools/analyze_rhino_data.py --videos-dir <path> --results-dir <path>
```

**Features:**
- **Coordinate System Analysis**: Verifies camera-to-world transformations
- **Dataset Statistics**:
  - Number of videos and frames
  - Annotation coverage (frames with/without annotations)
  - Bounding box size distributions
  - Confidence score distributions
- **Quality Checks**:
  - Missing files detection
  - Invalid bounding boxes
  - Behind-camera objects
  - Extreme dimensions
- **Video-Result Matching**:
  - Automatic pairing of video directories with result directories
  - Mismatch detection
  - Frame count comparison

**Example:**
```bash
python tools/analyze_rhino_data.py \
  --videos-dir examples/wd_data/rhinos_cami \
  --results-dir results \
  --output-report analysis_report.json
```

**Output:**
```
Dataset Analysis Report
======================

Videos Found: 12
Result Directories: 12
Matched Pairs: 12

Frame Statistics:
- Total frames: 1,800
- Frames with annotations: 1,650 (91.7%)
- Frames without annotations: 150 (8.3%)

Bounding Box Statistics:
- Total boxes: 1,650
- Mean dimensions: [2.5m, 1.8m, 4.2m]
- Mean distance from camera: 15.3m
- Confidence range: [0.65, 0.98]

Quality Issues:
- Behind camera: 0
- Extreme dimensions: 2
- Missing depth maps: 5

Video-Result Matching:
✓ rhin-105_1 → tmp-rhin-105_1-revisit-1
✓ rhin-30_3 → tmp-rhin-30_3-revisit-1
...
```

---

## Common Workflows

### Initial Data Exploration

When you first get CUT3R results, start with `explore_data.py`:

```bash
# 1. Explore the overall structure
python tools/explore_data.py --data-dir results/tmp-video1

# 2. Check a specific frame
python tools/explore_data.py --data-dir results/tmp-video1 --frame 001

# 3. Export structure to JSON
python tools/explore_data.py --data-dir results/tmp-video1 --output structure.json
```

### Dataset Validation

Before training, validate your dataset with `analyze_rhino_data.py`:

```bash
# 1. Run full analysis
python tools/analyze_rhino_data.py \
  --videos-dir data/videos \
  --results-dir results \
  --output-report validation.json

# 2. Check for issues
python tools/analyze_rhino_data.py \
  --videos-dir data/videos \
  --results-dir results \
  --check-quality \
  --min-confidence 0.7

# 3. Verify coordinate systems
python tools/analyze_rhino_data.py \
  --videos-dir data/videos \
  --results-dir results \
  --verify-coordinates
```

### Debugging Failed Conversions

If data conversion fails, use these tools to diagnose:

```bash
# 1. Check data structure
python tools/explore_data.py --data-dir results/failed_video

# 2. Verify video-result matching
python tools/analyze_rhino_data.py \
  --videos-dir videos \
  --results-dir results \
  --verbose

# 3. Look for missing files
python tools/analyze_rhino_data.py \
  --videos-dir videos \
  --results-dir results \
  --check-missing
```

---

## Notes

- Both tools generate detailed logs that can be redirected to files
- Use `--verbose` flag for additional debugging information
- Tools are read-only and safe to run on production data
- Large datasets may take several minutes to analyze
- Consider using `--sample` flag to analyze a subset for quick checks

## Troubleshooting

**Import errors:**
```bash
# Make sure you're in the CUT3R root directory
cd /path/to/CUT3R
python tools/explore_data.py --data-dir results
```

**Permission denied:**
```bash
# Check file permissions
ls -la results/
chmod +r results/* -R
```

**Memory issues with large datasets:**
```bash
# Use sampling for large datasets
python tools/explore_data.py --data-dir results --sample 100
```

## Output Formats

Both tools support multiple output formats:
- **Console**: Human-readable text output (default)
- **JSON**: Machine-readable structured data (`--output-format json`)
- **CSV**: Tabular data for spreadsheets (`--output-format csv`)
- **HTML**: Interactive reports (`--output-format html`)

Example:
```bash
python tools/analyze_rhino_data.py \
  --videos-dir videos \
  --results-dir results \
  --output-format html \
  --output report.html
```
