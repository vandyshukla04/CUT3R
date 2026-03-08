#!/bin/bash

# Launch labeling tool for existing elephant-4 results
# This uses your old demo_masks.py results - fully compatible!

set -e

echo "=================================================="
echo "Launching Labeling Tool for Elephant-4 Sequence"
echo "=================================================="
echo ""

# Paths for elep-5
AUTO_BBOXES="/home/shuklva/CUT3R/results/tmp-elep-5-revisit-1-vis/bounding_boxes"
IMAGES="/home/shuklva/CUT3R/examples/wd_data/elephants/elep-5"
OUTPUT="/home/shuklva/CUT3R/corrected_labels/elep-5"

# Check paths exist
if [ ! -d "$AUTO_BBOXES" ]; then
    echo "ERROR: Auto bboxes not found at $AUTO_BBOXES"
    exit 1
fi

if [ ! -d "$IMAGES" ]; then
    echo "ERROR: Images not found at $IMAGES"
    exit 1
fi

# Create output dir
mkdir -p "$OUTPUT"

# Count data
NUM_BBOXES=$(ls -1 "$AUTO_BBOXES"/*.json 2>/dev/null | wc -l)
NUM_IMAGES=$(ls -1 "$IMAGES"/*.jpg 2>/dev/null | wc -l)

echo "✓ Found $NUM_BBOXES bbox files"
echo "✓ Found $NUM_IMAGES images"
echo ""
echo "Output will be saved to: $OUTPUT"
echo ""

# Show tracking summary
if [ -f "/home/shuklva/CUT3R/results/tmp-elep-5-revisit-1/tracking_summary.json" ]; then
    echo "Tracking Summary:"
    python3 -c "
import json
with open('/home/shuklva/CUT3R/results/tmp-elep-5-revisit-1/tracking_summary.json') as f:
    data = json.load(f)
    print(f'  Total tracks: {len(data[\"tracks\"])}')
    for track in data['tracks']:
        print(f'    Track {track[\"track_id\"]}: {track[\"class_name\"]} - frames {track[\"first_frame\"]}-{track[\"last_frame\"]} ({track[\"detection_count\"]} detections)')
" 2>/dev/null || echo "  (Could not read tracking summary)"
    echo ""
fi

echo "=================================================="
echo "Browser will open at: http://localhost:8080"
echo ""
echo "QUICK WORKFLOW:"
echo "  1. Select track from dropdown"
echo "  2. Navigate frames, adjust bboxes"
echo "  3. Use proportion snapping (Species: Elephant)"
echo "  4. Mark keyframes (K) and interpolate"
echo "  5. Save (Ctrl+S)"
echo ""
echo "Press Ctrl+C to exit and save"
echo "=================================================="
echo ""

# Launch tool
python annotator_tool.py \
    --auto_bboxes "$AUTO_BBOXES" \
    --images "$IMAGES" \
    --output "$OUTPUT" \
    --port 8080

echo ""
echo "=================================================="
echo "Labeling session complete!"
echo "=================================================="
echo ""
echo "Check output:"
echo "  $OUTPUT/corrections.json"
echo "  $OUTPUT/corrected_bboxes/*.json"
echo "  $OUTPUT/annotation_statistics.json"
echo ""
