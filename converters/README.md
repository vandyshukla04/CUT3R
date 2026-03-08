# Data Converters

This directory contains tools for converting CUT3R wildlife detection outputs to various standard 3D detection formats.

## Unified Data Converter

The `unified_data_converter.py` script provides a single interface for converting to multiple formats.

### Supported Formats

1. **KITTI/MMDetection3D** - Standard format for 3D object detection
2. **Omni3D** - Format for Cube R-CNN and other Omni3D models
3. **Wildlife Info** - Pickle files for MMDetection3D training

### Usage

#### Convert to KITTI Format

```bash
python converters/unified_data_converter.py --format kitti \
  --videos-root /path/to/videos \
  --results-root /path/to/results \
  --output-dir /path/to/output \
  --target-class rhino
```

**Output Structure:**
```
output/
├── ImageSets/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── training/
│   ├── image_2/     # Resized images
│   ├── label_2/     # KITTI format labels
│   ├── calib/       # Camera calibration files
│   └── depth_maps/  # Depth maps (optional)
└── validation/
    └── ...
```

#### Convert to Omni3D Format

```bash
python converters/unified_data_converter.py --format omni3d \
  --videos-root /path/to/videos \
  --results-root /path/to/results \
  --output-dir /path/to/omni3d_output \
  --omni3d-image-output /path/to/images \
  --target-class rhino
```

**Output:**
- `RHINO_train.json` - Training set annotations
- `RHINO_val.json` - Validation set annotations
- `RHINO_test.json` - Test set annotations
- Images copied to `--omni3d-image-output` directory

#### Generate Wildlife Info Files

```bash
# First convert to KITTI, then generate info files
python converters/unified_data_converter.py --format wildlife_info \
  --output-dir /path/to/kitti_output
```

**Output:**
- `wildlife_infos_train.pkl`
- `wildlife_infos_val.pkl`
- `wildlife_infos_trainval.pkl`

### Configuration File

You can also use a YAML configuration file:

```yaml
# config.yaml
videos_root: /path/to/videos
results_root: /path/to/results
output_dir: /path/to/output
format: kitti
target_class: rhino
target_width: 512
target_height: 288
train_split_ratio: 0.8
min_confidence: 0.5
```

```bash
python converters/unified_data_converter.py --config config.yaml
```

### Input Data Structure

The converter expects the following input structure:

**Videos Directory:**
```
videos_root/
├── video1/
│   ├── frame_001.jpg
│   ├── frame_002.jpg
│   └── grounded-sam/
│       ├── frame_001_results.json
│       └── frame_002_results.json
└── video2/
    └── ...
```

**Results Directory:**
```
results_root/
├── tmp-video1-revisit-1/
│   ├── bounding_boxes/
│   │   ├── frame_001.json
│   │   └── frame_002.json
│   ├── camera/
│   │   └── camera_params.npz
│   └── depth/
│       ├── frame_001.npy
│       └── frame_002.npy
└── tmp-video2-revisit-1/
    └── ...
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--format` | Output format (kitti\|omni3d\|wildlife_info) | Required |
| `--videos-root` | Root directory containing video folders | Required* |
| `--results-root` | Root directory containing results folders | Required* |
| `--output-dir` | Output directory for converted data | Required |
| `--target-class` | Target class name | rhino |
| `--target-width` | Target image width | 512 |
| `--target-height` | Target image height | 288 |
| `--train-split-ratio` | Train/val split ratio | 0.8 |
| `--min-confidence` | Minimum confidence threshold | 0.5 |
| `--omni3d-image-output` | Output directory for Omni3D images | output_dir/images |

*Not required for `wildlife_info` format

### Data Splits

- **KITTI**: Random 80/20 train/val split (configurable)
- **Omni3D**: 60/20/20 train/val/test split by video

### Coordinate Systems

- **KITTI**: Camera coordinate system
  - X: right, Y: down, Z: forward
  - Rotation around Y-axis (yaw angle)
  - Dimensions: [height, width, length]

- **Omni3D**: Camera coordinate system
  - 3D boxes defined by center, dimensions, and rotation matrix
  - 2D boxes in both projected and SAM-tight formats

### Notes

- The converter automatically matches video directories with corresponding result directories
- Images are resized to target resolution for KITTI format
- Depth maps are preserved when available
- Camera intrinsics are included in all formats
- 2D bounding boxes from Grounded-SAM are used when available

### Examples

**Convert rhino data to KITTI:**
```bash
python converters/unified_data_converter.py --format kitti \
  --videos-root examples/wd_data/rhinos_cami \
  --results-root results \
  --output-dir data/rhino_kitti \
  --target-class rhino
```

**Convert zebra data to Omni3D:**
```bash
python converters/unified_data_converter.py --format omni3d \
  --videos-root examples/wd_data/zebras \
  --results-root results \
  --output-dir data/omni3d \
  --omni3d-image-output data/omni3d_images \
  --target-class zebra
```

### Troubleshooting

**No video folders found:**
- Check that `videos_root` contains subdirectories with .jpg files
- Verify file permissions

**No matching results folders:**
- Results folders should be named `tmp-{video_name}-revisit-*`
- Check `results_root` directory

**Missing annotations:**
- Ensure `bounding_boxes/` directory exists with .json files
- Check that frames have corresponding annotations

### Consolidated From

This unified converter consolidates functionality from:
- `prepare_data.py` - KITTI format conversion
- `rhino_to_omni3d.py` - Omni3D conversion
- `complete_conversion.py` - Complete pipeline
- `create_wildlife_info.py` - Wildlife info generation
