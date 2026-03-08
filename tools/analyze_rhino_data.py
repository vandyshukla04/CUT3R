#!/usr/bin/env python3
"""
Script to properly explore the rhino dataset structure
"""

import os
import json
import numpy as np
from pathlib import Path
import glob
from collections import defaultdict

def explore_directory_structure():
    """Map out the actual directory structure and matching"""
    
    video_base = "/home/shuklva/CUT3R/examples/wd_data/rhinos_cami"
    results_base = "/home/shuklva/CUT3R/results"
    
    print("="*70)
    print("DIRECTORY STRUCTURE EXPLORATION")
    print("="*70)
    
    # Find all video directories
    video_dirs = sorted(glob.glob(os.path.join(video_base, "rhin-*")))
    print(f"\nFound {len(video_dirs)} video directories:")
    
    video_mapping = {}
    for vdir in video_dirs:
        basename = os.path.basename(vdir)
        # Extract number from rhin-XX_Y format
        video_id = basename.replace("rhin-", "")
        video_mapping[video_id] = vdir
        print(f"  {basename} -> ID: {video_id}")
    
    # Find all result directories
    result_dirs = sorted(glob.glob(os.path.join(results_base, "tmp-rhin-*")))
    print(f"\nFound {len(result_dirs)} result directories:")
    
    result_mapping = {}
    for rdir in result_dirs:
        basename = os.path.basename(rdir)
        # Extract number from tmp-rhin-XX_Y-revisit-Z format
        parts = basename.replace("tmp-rhin-", "").split("-revisit-")
        result_id = parts[0]
        result_mapping[result_id] = rdir
        print(f"  {basename} -> ID: {result_id}")
    
    # Find matches
    print("\n" + "="*70)
    print("MATCHED PAIRS")
    print("="*70)
    
    matched_pairs = []
    for vid_id in video_mapping:
        if vid_id in result_mapping:
            matched_pairs.append((vid_id, video_mapping[vid_id], result_mapping[vid_id]))
            print(f"  {vid_id}: video and results found")
        else:
            print(f"  {vid_id}: video found but NO results")
    
    print(f"\nTotal matched pairs: {len(matched_pairs)}")
    
    return matched_pairs

def analyze_camera_structure(matched_pairs):
    """Analyze camera calibration structure across matched pairs"""
    
    print("\n" + "="*70)
    print("CAMERA CALIBRATION STRUCTURE ANALYSIS")
    print("="*70)
    
    if not matched_pairs:
        print("No matched pairs to analyze")
        return
    
    # Check first pair in detail
    first_id, video_dir, result_dir = matched_pairs[0]
    print(f"\nAnalyzing first matched pair: {first_id}")
    
    # Look for camera files in the result directory
    camera_dir = os.path.join(result_dir, "camera")
    
    if not os.path.exists(camera_dir):
        print(f"  ERROR: Camera directory not found at {camera_dir}")
        return
    
    # List camera files
    camera_files = sorted(glob.glob(os.path.join(camera_dir, "*")))
    print(f"  Found {len(camera_files)} files in camera directory")
    
    if camera_files:
        # Show file extensions
        extensions = defaultdict(int)
        for f in camera_files:
            ext = os.path.splitext(f)[1]
            extensions[ext] += 1
        
        print("  File types:")
        for ext, count in extensions.items():
            print(f"    {ext}: {count} files")
        
        # Analyze first camera file
        sample_file = camera_files[0]
        print(f"\n  Analyzing sample: {os.path.basename(sample_file)}")
        
        ext = os.path.splitext(sample_file)[1]
        
        if ext == '.npz':
            data = np.load(sample_file)
            print("  NPZ file keys:")
            for key in data.files:
                arr = data[key]
                print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}")
                if arr.shape == (3, 3):
                    print(f"      Possible intrinsic matrix K:")
                    print(f"      [[{arr[0,0]:.2f}, {arr[0,1]:.2f}, {arr[0,2]:.2f}],")
                    print(f"       [{arr[1,0]:.2f}, {arr[1,1]:.2f}, {arr[1,2]:.2f}],")
                    print(f"       [{arr[2,0]:.2f}, {arr[2,1]:.2f}, {arr[2,2]:.2f}]]")
            data.close()
            
        elif ext == '.json':
            with open(sample_file, 'r') as f:
                data = json.load(f)
            print(f"  JSON structure: {type(data)}")
            if isinstance(data, dict):
                print("  JSON keys:", list(data.keys())[:5], "...")
                
        elif ext == '.txt':
            with open(sample_file, 'r') as f:
                lines = f.readlines()
            print(f"  Text file with {len(lines)} lines")
            if lines:
                print(f"  First line: {lines[0].strip()}")
    
    # Check if camera files are per-frame or per-video
    print("\n" + "="*70)
    print("CAMERA CALIBRATION SCOPE")
    print("="*70)
    
    # Count camera files vs frame files
    for vid_id, video_dir, result_dir in matched_pairs[:3]:  # Check first 3
        camera_dir = os.path.join(result_dir, "camera")
        bbox_dir = os.path.join(result_dir, "bounding_boxes")
        video_frames = glob.glob(os.path.join(video_dir, "*.jpg"))
        
        if os.path.exists(camera_dir):
            camera_files = glob.glob(os.path.join(camera_dir, "*"))
            bbox_files = glob.glob(os.path.join(bbox_dir, "*.json")) if os.path.exists(bbox_dir) else []
            
            print(f"\n  {vid_id}:")
            print(f"    Video frames: {len(video_frames)}")
            print(f"    Camera files: {len(camera_files)}")
            print(f"    BBox files: {len(bbox_files)}")
            
            if len(camera_files) == len(bbox_files):
                print("    -> Likely per-frame calibration")
            elif len(camera_files) == 1:
                print("    -> Likely single calibration for video")
            else:
                print("    -> Unclear calibration structure")

def count_total_annotations(matched_pairs):
    """Count total frames with 3D annotations"""
    
    print("\n" + "="*70)
    print("ANNOTATION STATISTICS")
    print("="*70)
    
    total_frames = 0
    total_annotated = 0
    
    for vid_id, video_dir, result_dir in matched_pairs:
        video_frames = glob.glob(os.path.join(video_dir, "*.jpg"))
        bbox_dir = os.path.join(result_dir, "bounding_boxes")
        
        if os.path.exists(bbox_dir):
            bbox_files = glob.glob(os.path.join(bbox_dir, "*.json"))
            total_frames += len(video_frames)
            total_annotated += len(bbox_files)
    
    print(f"\nTotal video frames: {total_frames}")
    print(f"Total frames with 3D annotations: {total_annotated}")
    print(f"Annotation coverage: {100*total_annotated/total_frames:.1f}%")
    
    print("\nSuggested data splits:")
    print(f"  Train (70%): ~{int(0.7*total_annotated)} frames")
    print(f"  Val (15%): ~{int(0.15*total_annotated)} frames")
    print(f"  Test (15%): ~{int(0.15*total_annotated)} frames")

def check_config_files():
    """Look for Base.yaml and other config files"""
    
    print("\n" + "="*70)
    print("CONFIG FILE CHECK")
    print("="*70)
    
    config_paths = [
        "/home/shuklva/ovmono3d/configs",
        "/home/shuklva/CUT3R/cubercnn/configs",
        "/home/shuklva/CUT3R/configs",
        "/home/shuklva/OVMONO3D/configs",
        "configs"  # relative path
    ]
    
    for path in config_paths:
        if os.path.exists(path):
            print(f"\nChecking {path}:")
            yaml_files = glob.glob(os.path.join(path, "*.yaml"))
            yaml_files.extend(glob.glob(os.path.join(path, "*.yml")))
            
            if yaml_files:
                print(f"  Found {len(yaml_files)} YAML files:")
                for f in yaml_files[:10]:  # Show first 10
                    print(f"    - {os.path.basename(f)}")
                    if "Base" in os.path.basename(f):
                        print("      ^^ BASE CONFIG FOUND!")
            else:
                print("  No YAML files found")
        else:
            print(f"\n{path} does not exist")

if __name__ == "__main__":
    # Explore structure
    matched_pairs = explore_directory_structure()
    
    # Analyze camera files
    analyze_camera_structure(matched_pairs)
    
    # Count annotations
    count_total_annotations(matched_pairs)
    
    # Check for config files
    check_config_files()