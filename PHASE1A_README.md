# Phase 1A: Species-Constrained Bounding Box Fitting

## Overview

This branch (`phase1a-species-dimensions`) implements species-specific dimension constraints for 3D bounding box fitting to address bbox size inconsistencies across different viewpoints.

## Problem Addressed

**Before Phase 1A:**
- Bounding boxes fit purely using PCA on visible points
- Box size varies dramatically based on viewpoint:
  - Side view: Sees full animal → larger box
  - Top-down view: Sees only dorsal surface → small/flat box
  - Front view: Foreshortening → compressed box
- No use of known animal dimensions

**After Phase 1A:**
- Bounding boxes constrained by known species dimensions
- More consistent sizes across viewpoints
- Automatic scale estimation from visible points
- Realistic animal proportions maintained

## Changes Made

### 1. Added Species Dimensions Database

**File:** `demo_masks.py` (lines 998-1032)

```python
self.animal_dimensions = {
    'zebra': {'length': 2.5, 'width': 0.7, 'height': 1.4},
    'rhino': {'length': 3.8, 'width': 1.5, 'height': 1.8},
    'rhinoceros': {'length': 3.8, 'width': 1.5, 'height': 1.8},
    'elephant': {'length': 5.5, 'width': 2.5, 'height': 3.2},
    'animal': {'length': 2.0, 'width': 0.6, 'height': 1.2}  # Generic fallback
}
```

**Dimensions are in meters:**
- `length`: Nose to tail
- `width`: Side to side (at widest point)
- `height`: Ground to top of back

### 2. New Methods Added

#### `get_species_dimensions(class_name)` (lines 1066-1083)
- Looks up species-specific dimensions
- Falls back to generic quadruped for unknown species

#### `compute_species_constrained_bbox(points_3d, class_name, ground_normal)` (lines 1085-1167)
- Computes PCA bbox as before
- Estimates scale from ratio of PCA dimensions to species dimensions
- Blends PCA and species dimensions based on `species_constraint_strength`
- Adjusts center height if ground plane is available

### 3. Modified Bbox Computation Call

**File:** `demo_masks.py` (lines 1437-1458)

**Before:**
```python
bbox_result = self.compute_oriented_bbox_pca(filtered_points)
```

**After:**
```python
bbox_result = self.compute_species_constrained_bbox(
    filtered_points,
    class_name,
    ground_normal=ground_normal
)
```

## Configuration Parameters

### Enable/Disable Species Constraints

**File:** `demo_masks.py` (lines 1031-1032)

```python
self.use_species_constraints = True  # Set to False to revert to old behavior
self.species_constraint_strength = 0.7  # How much to trust species dims (0-1)
```

**`use_species_constraints`:**
- `True` (default): Use species-constrained fitting
- `False`: Revert to pure PCA fitting (old behavior)

**`species_constraint_strength`:**
- `0.0`: Use only PCA dimensions (ignores species constraints)
- `1.0`: Use only species dimensions (ignores PCA)
- `0.7` (default): Blend 70% species dimensions + 30% PCA dimensions

## How It Works

### Algorithm:

1. **PCA Fitting:**
   - Compute principal axes and bounding box from visible points
   - Get initial orientation and approximate dimensions

2. **Scale Estimation:**
   - Compare PCA dimensions to known species dimensions
   - Compute scale factor: `scale = median(PCA_dims / species_dims)`
   - Use median for robustness to outliers

3. **Dimension Blending:**
   - Scale species dimensions: `species_scaled = species_dims × scale`
   - Blend: `final_dims = α × species_scaled + (1-α) × PCA_dims`
   - Where α = `species_constraint_strength`

4. **Ground Constraint (if available):**
   - Find minimum point height relative to ground
   - Adjust bbox center so bottom touches ground

### Example:

**Input:**
- Animal: zebra
- PCA dimensions: [1.8, 0.5, 1.0] meters (partial view from top)
- Species dimensions: [2.5, 0.7, 1.4] meters

**Processing:**
- Scale factors: [1.8/2.5, 0.5/0.7, 1.0/1.4] = [0.72, 0.71, 0.71]
- Median scale: 0.71
- Species scaled: [1.78, 0.50, 0.99] meters
- Final (α=0.7): 0.7×[1.78,0.50,0.99] + 0.3×[1.8,0.5,1.0] = [1.79, 0.50, 0.99]

**Result:** More consistent dimensions despite partial visibility

## How to Revert to Old Behavior

### Option 1: Quick Toggle (No Code Changes)

**Edit** `demo_masks.py` line 1031:
```python
self.use_species_constraints = False  # Disables species constraints
```

Run your inference as normal. This reverts to pure PCA fitting.

### Option 2: Switch Git Branch

```bash
# Save any uncommitted changes
git stash

# Switch back to main branch
git checkout main

# Or switch to vandi_v1 branch
git checkout vandi_v1

# Restore uncommitted changes if needed
git stash pop
```

### Option 3: Remove Phase 1A Code Entirely

```bash
# Create backup of current branch
git branch phase1a-backup

# Reset to commit before Phase 1A
git reset --hard <commit-hash-before-phase1a>
```

## Testing Phase 1A

### Run on Your Data:

```bash
python demo_masks.py \
    --model_path src/cut3r_512_dpt_4_64.pth \
    --seq_path examples/your_sequence \
    --mask_dir path/to/masks \
    --output_dir results/phase1a_test \
    --size 512
```

### Check Output:

1. **Console Output:**
   - Look for lines starting with `📏 Species:`
   - Shows scale factor and dimension comparison

2. **Saved Bboxes:**
   - `results/phase1a_test/bounding_boxes/*.json`
   - Check `dimensions` field for each bbox

3. **Visual Inspection:**
   - Compare bbox sizes across different frames
   - Check if sizes are more consistent than before

### Expected Improvements:

✅ **More consistent bbox sizes** across viewpoints
✅ **Realistic dimensions** (zebra ≈ 2.5m, rhino ≈ 3.8m, elephant ≈ 5.5m)
✅ **Less sensitivity** to partial views (top-down, front, etc.)
✅ **Automatic fallback** for unknown species (uses generic quadruped)

### Potential Issues:

⚠️ **If scale estimation is bad** (e.g., very partial view):
- Try lowering `species_constraint_strength` to 0.5 or 0.3
- This trusts PCA more than species dimensions

⚠️ **If animal dimensions are wrong** (different subspecies):
- Update dimensions in `self.animal_dimensions` dictionary
- Or use generic fallback

## Adding New Species

To add support for new animals:

**Edit** `demo_masks.py` (lines 1001-1028):

```python
self.animal_dimensions = {
    # ... existing animals ...

    # Add your new species
    'giraffe': {
        'length': 4.5,   # Body length in meters
        'width': 1.2,    # Body width in meters
        'height': 5.5,   # Height to top of back (not including neck!)
    },
}
```

**Also add color** (optional, lines 982-992):

```python
self.class_colors = {
    # ... existing colors ...
    'giraffe': np.array([0.9, 0.8, 0.4]),  # Sandy yellow
}
```

## Next Steps: Phase 1B & 1C

**Phase 1B:** Temporal point accumulation
- Accumulate points from nearby frames (±2-5 frames)
- Get more complete 3D model of animal
- Further reduce bbox flickering

**Phase 1C:** Temporal smoothing
- Apply Kalman filter or exponential moving average
- Smooth bbox position, size, orientation over time
- Final reduction of jitter

## Contact & Troubleshooting

**Branch:** `phase1a-species-dimensions`

**To report issues:**
1. Check if `use_species_constraints = False` fixes it (narrows down cause)
2. Share console output (especially `📏 Species:` lines)
3. Share sample bbox JSON before/after Phase 1A

**Quick Debug:**
```python
# In demo_masks.py, after line 1163, add:
print(f"    🐾 PCA scale factors: {scale_factors}")
print(f"    🐾 Final bbox center: {center_constrained}")
```

This helps diagnose scale estimation issues.
