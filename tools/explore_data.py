#!/usr/bin/env python3
"""
Analysis tool for understanding your 3D reconstruction outputs and converting to Cube R-CNN format.
This script analyzes the outputs from your enhanced_demo.py and provides conversion utilities.
"""

import os
import json
import numpy as np
import glob
from collections import defaultdict, OrderedDict
from typing import Dict, List, Any, Tuple, Optional
import argparse


class OutputAnalyzer:
    """Analyzes outputs from your 3D reconstruction pipeline."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.analysis_results = {}
        
    def analyze_directory_structure(self):
        """Analyze the directory structure and file types."""
        print(f"\n📁 === DIRECTORY STRUCTURE ANALYSIS ===")
        print(f"Output directory: {self.output_dir}")
        
        structure = {}
        for root, dirs, files in os.walk(self.output_dir):
            rel_path = os.path.relpath(root, self.output_dir)
            if rel_path == '.':
                rel_path = 'root'
            
            file_types = defaultdict(int)
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                file_types[ext] += 1
            
            structure[rel_path] = {
                'subdirs': dirs,
                'file_types': dict(file_types),
                'total_files': len(files)
            }
        
        for path, info in structure.items():
            print(f"\n📂 {path}/")
            if info['subdirs']:
                print(f"  Subdirectories: {info['subdirs']}")
            print(f"  Files: {info['total_files']}")
            for ext, count in info['file_types'].items():
                print(f"    {ext}: {count} files")
        
        self.analysis_results['directory_structure'] = structure
        return structure

    def analyze_camera_data(self):
        """Analyze camera pose and intrinsic data."""
        print(f"\n📷 === CAMERA DATA ANALYSIS ===")
        
        camera_dir = os.path.join(self.output_dir, 'camera')
        if not os.path.exists(camera_dir):
            print("❌ No camera directory found")
            return None
        
        camera_files = glob.glob(os.path.join(camera_dir, '*.npz'))
        print(f"Found {len(camera_files)} camera files")
        
        if not camera_files:
            return None
        
        # Analyze first few camera files
        sample_files = sorted(camera_files)[:3]
        camera_analysis = {
            'total_files': len(camera_files),
            'sample_data': []
        }
        
        for cam_file in sample_files:
            frame_name = os.path.splitext(os.path.basename(cam_file))[0]
            data = np.load(cam_file)
            
            sample_info = {
                'frame': frame_name,
                'keys': list(data.keys()),
                'pose_shape': data['pose'].shape if 'pose' in data else None,
                'intrinsics_shape': data['intrinsics'].shape if 'intrinsics' in data else None,
            }
            
            if 'pose' in data:
                pose = data['pose']
                sample_info['pose_sample'] = {
                    'rotation': pose[:3, :3].tolist(),
                    'translation': pose[:3, 3].tolist(),
                    'determinant': float(np.linalg.det(pose[:3, :3]))
                }
            
            if 'intrinsics' in data:
                K = data['intrinsics']
                sample_info['intrinsics_sample'] = {
                    'fx': float(K[0, 0]),
                    'fy': float(K[1, 1]),
                    'cx': float(K[0, 2]),
                    'cy': float(K[1, 2]),
                    'matrix': K.tolist()
                }
            
            camera_analysis['sample_data'].append(sample_info)
            
            print(f"  📸 {frame_name}:")
            print(f"    Keys: {list(data.keys())}")
            if 'pose' in data:
                print(f"    Pose: {pose[:3, :3].shape} rotation + {pose[:3, 3].shape} translation")
                print(f"    Translation: [{pose[0,3]:.3f}, {pose[1,3]:.3f}, {pose[2,3]:.3f}]")
            if 'intrinsics' in data:
                print(f"    Intrinsics: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
        
        self.analysis_results['camera_data'] = camera_analysis
        return camera_analysis

    def analyze_bounding_boxes(self):
        """Analyze 3D bounding box data with tracking information."""
        print(f"\n📦 === BOUNDING BOX ANALYSIS ===")
        
        bbox_dir = os.path.join(self.output_dir, 'bounding_boxes')
        if not os.path.exists(bbox_dir):
            print("❌ No bounding_boxes directory found")
            return None
        
        bbox_files = glob.glob(os.path.join(bbox_dir, '*.json'))
        print(f"Found {len(bbox_files)} bounding box files")
        
        if not bbox_files:
            return None
        
        bbox_analysis = {
            'total_files': len(bbox_files),
            'total_boxes': 0,
            'classes': set(),
            'track_ids': set(),
            'sample_data': [],
            'class_distribution': defaultdict(int),
            'tracking_stats': {
                'tracked_boxes': 0,
                'untracked_boxes': 0,
                'unique_tracks': set()
            }
        }
        
        # Analyze all bbox files
        for bbox_file in sorted(bbox_files):
            frame_name = os.path.splitext(os.path.basename(bbox_file))[0]
            
            with open(bbox_file, 'r') as f:
                frame_bboxes = json.load(f)
            
            bbox_analysis['total_boxes'] += len(frame_bboxes)
            
            for bbox in frame_bboxes:
                # Class information
                class_name = bbox['class_name']
                bbox_analysis['classes'].add(class_name)
                bbox_analysis['class_distribution'][class_name] += 1
                
                # Tracking information
                track_id = bbox.get('track_id', -1)
                if track_id >= 0:
                    bbox_analysis['track_ids'].add(track_id)
                    bbox_analysis['tracking_stats']['tracked_boxes'] += 1
                    bbox_analysis['tracking_stats']['unique_tracks'].add(track_id)
                else:
                    bbox_analysis['tracking_stats']['untracked_boxes'] += 1
        
        # Sample data from first few files
        sample_files = sorted(bbox_files)[:3]
        for bbox_file in sample_files:
            frame_name = os.path.splitext(os.path.basename(bbox_file))[0]
            
            with open(bbox_file, 'r') as f:
                frame_bboxes = json.load(f)
            
            sample_info = {
                'frame': frame_name,
                'num_boxes': len(frame_bboxes),
                'boxes': []
            }
            
            for bbox in frame_bboxes:
                box_info = {
                    'class_name': bbox['class_name'],
                    'confidence': bbox['confidence'],
                    'instance_id': bbox['instance_id'],
                    'track_id': bbox.get('track_id', -1),
                    'center': bbox['center'],
                    'dimensions': bbox['dimensions'],
                    'has_rotation': 'rotation_matrix' in bbox and bbox['rotation_matrix'] is not None
                }
                sample_info['boxes'].append(box_info)
            
            bbox_analysis['sample_data'].append(sample_info)
        
        # Convert sets to lists for JSON serialization
        bbox_analysis['classes'] = sorted(list(bbox_analysis['classes']))
        bbox_analysis['track_ids'] = sorted(list(bbox_analysis['track_ids']))
        bbox_analysis['tracking_stats']['unique_tracks'] = len(bbox_analysis['tracking_stats']['unique_tracks'])
        
        print(f"  📊 Total bounding boxes: {bbox_analysis['total_boxes']}")
        print(f"  📋 Classes found: {bbox_analysis['classes']}")
        print(f"  🏷️ Unique track IDs: {len(bbox_analysis['track_ids'])}")
        print(f"  🔗 Tracked boxes: {bbox_analysis['tracking_stats']['tracked_boxes']}")
        print(f"  ❓ Untracked boxes: {bbox_analysis['tracking_stats']['untracked_boxes']}")
        print(f"  📈 Class distribution:")
        for class_name, count in bbox_analysis['class_distribution'].items():
            print(f"    {class_name}: {count}")
        
        # Show sample data
        for sample in bbox_analysis['sample_data']:
            print(f"\n  🔍 Sample frame {sample['frame']}: {sample['num_boxes']} boxes")
            for box in sample['boxes'][:2]:  # Show first 2 boxes
                print(f"    - {box['class_name']} (track {box['track_id']}, conf {box['confidence']:.3f})")
                print(f"      Center: [{box['center'][0]:.3f}, {box['center'][1]:.3f}, {box['center'][2]:.3f}]")
                print(f"      Dims: [{box['dimensions'][0]:.3f}, {box['dimensions'][1]:.3f}, {box['dimensions'][2]:.3f}]")
        
        self.analysis_results['bounding_boxes'] = bbox_analysis
        return bbox_analysis

    def analyze_instance_labels(self):
        """Analyze instance segmentation labels."""
        print(f"\n🎭 === INSTANCE LABELS ANALYSIS ===")
        
        labels_dir = os.path.join(self.output_dir, 'instance_labels')
        if not os.path.exists(labels_dir):
            print("❌ No instance_labels directory found")
            return None
        
        label_files = glob.glob(os.path.join(labels_dir, '*.npy'))
        print(f"Found {len(label_files)} instance label files")
        
        if not label_files:
            return None
        
        labels_analysis = {
            'total_files': len(label_files),
            'sample_data': []
        }
        
        # Analyze first few files
        sample_files = sorted(label_files)[:3]
        for label_file in sample_files:
            frame_name = os.path.splitext(os.path.basename(label_file))[0]
            labels = np.load(label_file)
            
            unique_labels = np.unique(labels)
            non_zero_labels = unique_labels[unique_labels > 0]
            
            sample_info = {
                'frame': frame_name,
                'shape': labels.shape,
                'dtype': str(labels.dtype),
                'unique_instances': len(non_zero_labels),
                'instance_ids': non_zero_labels.tolist(),
                'total_pixels': labels.size,
                'labeled_pixels': int(np.sum(labels > 0)),
                'background_pixels': int(np.sum(labels == 0))
            }
            
            coverage = (sample_info['labeled_pixels'] / sample_info['total_pixels']) * 100
            sample_info['coverage_percent'] = coverage
            
            labels_analysis['sample_data'].append(sample_info)
            
            print(f"  🎭 {frame_name}: {labels.shape}")
            print(f"    Instances: {len(non_zero_labels)} ({non_zero_labels.tolist()})")
            print(f"    Coverage: {coverage:.1f}% ({sample_info['labeled_pixels']:,} / {sample_info['total_pixels']:,} pixels)")
        
        self.analysis_results['instance_labels'] = labels_analysis
        return labels_analysis

    def analyze_tracking_summary(self):
        """Analyze tracking summary if available."""
        print(f"\n🔗 === TRACKING SUMMARY ANALYSIS ===")
        
        tracking_file = os.path.join(self.output_dir, 'tracking_summary.json')
        if not os.path.exists(tracking_file):
            print("❌ No tracking_summary.json found")
            return None
        
        with open(tracking_file, 'r') as f:
            tracking_data = json.load(f)
        
        print(f"  📊 Total tracks: {tracking_data['total_tracks']}")
        print(f"  🎬 Frames processed: {tracking_data['frames_processed']}")
        print(f"  📦 Total detections: {tracking_data['total_detections']}")
        print(f"  📏 Average tracklet length: {tracking_data['avg_tracklet_length']:.1f}")
        print(f"  📈 Max tracklet length: {tracking_data['max_tracklet_length']}")
        
        if 'class_statistics' in tracking_data:
            print(f"\n  📋 Per-class tracking:")
            for class_name, stats in tracking_data['class_statistics'].items():
                print(f"    {class_name}: {stats['track_count']} tracks, avg {stats['avg_track_length']:.1f} frames")
        
        if 'tracks' in tracking_data:
            print(f"\n  🔍 Sample tracks:")
            sample_tracks = list(tracking_data['tracks'].items())[:3]
            for track_id, track_info in sample_tracks:
                print(f"    Track {track_id} ({track_info['class_name']}): "
                      f"{track_info['length']} frames ({track_info['first_frame']}-{track_info['last_frame']})")
        
        self.analysis_results['tracking_summary'] = tracking_data
        return tracking_data

    def check_cube_rcnn_compatibility(self):
        """Check compatibility with Cube R-CNN format requirements."""
        print(f"\n🔧 === CUBE R-CNN COMPATIBILITY CHECK ===")
        
        compatibility = {
            'has_images': False,
            'has_camera_data': False,
            'has_3d_bboxes': False,
            'has_instance_masks': False,
            'missing_components': [],
            'conversion_needed': []
        }
        
        # Check for required components
        camera_data = self.analysis_results.get('camera_data')
        bbox_data = self.analysis_results.get('bounding_boxes')
        labels_data = self.analysis_results.get('instance_labels')
        
        if camera_data:
            compatibility['has_camera_data'] = True
            print("  ✅ Camera poses and intrinsics available")
        else:
            compatibility['missing_components'].append('camera_data')
            print("  ❌ Missing camera data")
        
        if bbox_data:
            compatibility['has_3d_bboxes'] = True
            print("  ✅ 3D bounding boxes available")
            
            # Check bbox format compatibility
            if bbox_data['sample_data']:
                sample_box = bbox_data['sample_data'][0]['boxes'][0]
                
                # Check required fields for Cube R-CNN
                required_fields = ['center', 'dimensions']  # rotation is optional
                missing_fields = []
                format_issues = []
                
                for field in required_fields:
                    if field not in sample_box:
                        missing_fields.append(field)
                
                if missing_fields:
                    compatibility['conversion_needed'].append(f"bbox_missing_fields: {missing_fields}")
                
                # Check coordinate system (Cube R-CNN expects camera coordinates)
                print("    📍 Coordinate system: Needs verification (camera vs world coords)")
                compatibility['conversion_needed'].append('verify_coordinate_system')
                
        else:
            compatibility['missing_components'].append('3d_bboxes')
            print("  ❌ Missing 3D bounding boxes")
        
        if labels_data:
            compatibility['has_instance_masks'] = True
            print("  ✅ Instance masks available")
        else:
            compatibility['missing_components'].append('instance_masks')
            print("  ❌ Missing instance masks")
        
        # Check for original images (needed for training)
        annotated_dir = os.path.join(self.output_dir, 'annotated_2d')
        if os.path.exists(annotated_dir):
            compatibility['has_images'] = True
            print("  ✅ Annotated images available")
        else:
            compatibility['missing_components'].append('original_images')
            print("  ❌ Missing original images")
        
        # Assess overall compatibility
        critical_missing = len([x for x in compatibility['missing_components'] 
                               if x in ['camera_data', '3d_bboxes']])
        
        if critical_missing == 0:
            print(f"\n  🎉 GOOD: Core components available for Cube R-CNN conversion!")
            if compatibility['conversion_needed']:
                print(f"  ⚙️ Conversion needed for: {compatibility['conversion_needed']}")
        else:
            print(f"\n  ⚠️ WARNING: Missing {critical_missing} critical components")
        
        self.analysis_results['compatibility'] = compatibility
        return compatibility

    def generate_conversion_recommendations(self):
        """Generate specific recommendations for converting to Cube R-CNN format."""
        print(f"\n💡 === CONVERSION RECOMMENDATIONS ===")
        
        compatibility = self.analysis_results.get('compatibility', {})
        
        recommendations = []
        
        if compatibility.get('has_camera_data') and compatibility.get('has_3d_bboxes'):
            recommendations.append({
                'priority': 'HIGH',
                'task': 'Create OMNI3D JSON format converter',
                'description': 'Convert your outputs to OMNI3D dataset JSON format',
                'details': [
                    'Combine camera data, bounding boxes, and images into single JSON',
                    'Map your class names to OMNI3D category IDs', 
                    'Convert coordinate systems if needed (world -> camera coordinates)',
                    'Create train/val/test splits'
                ]
            })
        
        if 'verify_coordinate_system' in compatibility.get('conversion_needed', []):
            recommendations.append({
                'priority': 'HIGH', 
                'task': 'Verify coordinate system compatibility',
                'description': 'Ensure bounding boxes are in camera coordinate system',
                'details': [
                    'Cube R-CNN expects boxes in camera coordinates',
                    'Your boxes might be in world coordinates',
                    'Transform boxes using inverse camera pose if needed',
                    'Verify box orientations match Cube R-CNN conventions'
                ]
            })
        
        if not compatibility.get('has_images'):
            recommendations.append({
                'priority': 'MEDIUM',
                'task': 'Collect original images',
                'description': 'Original RGB images are needed for training',
                'details': [
                    'Save original input images alongside annotations',
                    'Ensure images match the camera parameters',
                    'Consider image preprocessing (resize, normalize)'
                ]
            })
        
        recommendations.append({
            'priority': 'MEDIUM',
            'task': 'Create data validation pipeline', 
            'description': 'Validate converted data before training',
            'details': [
                'Check bbox projections align with 2D masks',
                'Verify camera calibration accuracy',
                'Test with small Cube R-CNN training run',
                'Compare with existing OMNI3D samples'
            ]
        })
        
        recommendations.append({
            'priority': 'LOW',
            'task': 'Optimize for Cube R-CNN performance',
            'description': 'Fine-tune data for best training results', 
            'details': [
                'Filter low-confidence detections',
                'Balance class distributions',
                'Add data augmentation compatible boxes',
                'Consider temporal consistency in tracking'
            ]
        })
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n  {i}. [{rec['priority']}] {rec['task']}")
            print(f"     {rec['description']}")
            for detail in rec['details']:
                print(f"     • {detail}")
        
        self.analysis_results['recommendations'] = recommendations
        return recommendations

    def save_analysis_report(self, output_file: str):
        """Save complete analysis report to JSON."""
        print(f"\n💾 Saving analysis report to {output_file}")
        
        with open(output_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        print(f"  ✅ Analysis report saved!")

    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run all analysis functions."""
        print(f"🔍 === STARTING COMPLETE OUTPUT ANALYSIS ===")
        print(f"Target directory: {self.output_dir}")
        
        # Run all analysis functions
        self.analyze_directory_structure()
        self.analyze_camera_data() 
        self.analyze_bounding_boxes()
        self.analyze_instance_labels()
        self.analyze_tracking_summary()
        self.check_cube_rcnn_compatibility()
        self.generate_conversion_recommendations()
        
        print(f"\n✅ === ANALYSIS COMPLETE ===")
        return self.analysis_results


class CubeRCNNConverter:
    """Converts your outputs to Cube R-CNN OMNI3D format."""
    
    def __init__(self, output_dir: str, original_images_dir: str = None):
        self.output_dir = output_dir
        self.original_images_dir = original_images_dir
        
    def load_all_data(self) -> Dict[str, Any]:
        """Load all data from your pipeline outputs."""
        print("📖 Loading all pipeline data...")
        
        data = {
            'cameras': {},
            'bboxes': {},
            'instance_labels': {},
            'images': {}
        }
        
        # Load camera data
        camera_dir = os.path.join(self.output_dir, 'camera')
        if os.path.exists(camera_dir):
            for cam_file in glob.glob(os.path.join(camera_dir, '*.npz')):
                frame_name = os.path.splitext(os.path.basename(cam_file))[0]
                data['cameras'][frame_name] = np.load(cam_file)
        
        # Load bounding boxes
        bbox_dir = os.path.join(self.output_dir, 'bounding_boxes')
        if os.path.exists(bbox_dir):
            for bbox_file in glob.glob(os.path.join(bbox_dir, '*.json')):
                frame_name = os.path.splitext(os.path.basename(bbox_file))[0]
                with open(bbox_file, 'r') as f:
                    data['bboxes'][frame_name] = json.load(f)
        
        # Load instance labels
        labels_dir = os.path.join(self.output_dir, 'instance_labels')
        if os.path.exists(labels_dir):
            for label_file in glob.glob(os.path.join(labels_dir, '*.npy')):
                frame_name = os.path.splitext(os.path.basename(label_file))[0]
                data['instance_labels'][frame_name] = np.load(label_file)
        
        print(f"  📷 Loaded {len(data['cameras'])} camera files")
        print(f"  📦 Loaded {len(data['bboxes'])} bbox files") 
        print(f"  🎭 Loaded {len(data['instance_labels'])} instance label files")
        
        return data

    def convert_to_omni3d_format(self, data: Dict[str, Any], output_json: str):
        """Convert loaded data to OMNI3D JSON format."""
        print(f"🔄 Converting to OMNI3D format...")
        
        # OMNI3D JSON structure
        omni3d_data = {
            "info": {
                "description": "Converted from enhanced_demo.py output",
                "version": "1.0",
                "year": 2024,
                "contributor": "Enhanced Demo Pipeline",
                "date_created": "2024-01-01"
            },
            "categories": [],
            "images": [],
            "annotations": []
        }
        
        # Extract unique classes and create category mapping
        all_classes = set()
        for frame_bboxes in data['bboxes'].values():
            for bbox in frame_bboxes:
                all_classes.add(bbox['class_name'])
        
        # Create category entries
        category_map = {}
        for i, class_name in enumerate(sorted(all_classes)):
            category_id = i + 1  # COCO format starts at 1
            omni3d_data["categories"].append({
                "id": category_id,
                "name": class_name,
                "supercategory": "object"
            })
            category_map[class_name] = category_id
        
        print(f"  📋 Created {len(category_map)} categories: {list(category_map.keys())}")
        
        # Process each frame
        image_id = 1
        annotation_id = 1
        
        for frame_name in sorted(data['cameras'].keys()):
            if frame_name not in data['bboxes']:
                continue
            
            # Image entry
            # Note: You'll need actual image dimensions here
            # This is a placeholder - you should read from actual images
            image_entry = {
                "id": image_id,
                "file_name": f"{frame_name}.jpg",  # Adjust extension as needed
                "width": 640,  # PLACEHOLDER - get from actual images
                "height": 480,  # PLACEHOLDER - get from actual images
            }
            
            # Add camera parameters
            if frame_name in data['cameras']:
                cam_data = data['cameras'][frame_name]
                pose = cam_data['pose']
                intrinsics = cam_data['intrinsics']
                
                # Convert pose to OMNI3D format (camera-to-world)
                image_entry['pose'] = pose.tolist()
                image_entry['intrinsics'] = intrinsics.tolist()
            
            omni3d_data["images"].append(image_entry)
            
            # Process annotations for this frame
            if frame_name in data['bboxes']:
                frame_bboxes = data['bboxes'][frame_name]
                
                for bbox in frame_bboxes:
                    # Convert your bbox format to OMNI3D format
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_map[bbox['class_name']],
                        "bbox_mode": "3d",  # OMNI3D specific
                        
                        # 3D bounding box parameters
                        "location": bbox['center'],  # [x, y, z] in camera coordinates
                        "dimensions": bbox['dimensions'],  # [l, w, h] 
                        "rotation_y": 0.0,  # You'll need to extract this from rotation_matrix
                        
                        # Additional info
                        "score": bbox['confidence'],
                        "area": float(np.prod(bbox['dimensions'][:2])),  # Approximate 2D area
                        "iscrowd": 0,
                        
                        # Your tracking info (not standard OMNI3D but can be useful)
                        "track_id": bbox.get('track_id', -1),
                        "instance_id": bbox['instance_id']
                    }
                    
                    # Convert rotation matrix to rotation_y if available
                    if 'rotation_matrix' in bbox and bbox['rotation_matrix']:
                        rot_matrix = np.array(bbox['rotation_matrix'])
                        # Extract yaw angle from rotation matrix
                        # This is a simplified conversion - you may need more sophisticated handling
                        rotation_y = float(np.arctan2(rot_matrix[2, 0], rot_matrix[0, 0]))
                        annotation['rotation_y'] = rotation_y
                    
                    omni3d_data["annotations"].append(annotation)
                    annotation_id += 1
            
            image_id += 1
        
        # Save converted data
        with open(output_json, 'w') as f:
            json.dump(omni3d_data, f, indent=2)
        
        print(f"  ✅ Saved OMNI3D format to {output_json}")
        print(f"  📊 {len(omni3d_data['images'])} images, {len(omni3d_data['annotations'])} annotations")
        
        return omni3d_data

    def validate_conversion(self, omni3d_json: str):
        """Validate the converted OMNI3D format."""
        print(f"🔍 Validating conversion...")
        
        with open(omni3d_json, 'r') as f:
            data = json.load(f)
        
        # Basic validation checks
        required_keys = ['info', 'categories', 'images', 'annotations']
        for key in required_keys:
            if key not in data:
                print(f"  ❌ Missing required key: {key}")
                return False
        
        # Check data consistency
        image_ids = {img['id'] for img in data['images']}
        category_ids = {cat['id'] for cat in data['categories']}
        
        invalid_refs = 0
        for ann in data['annotations']:
            if ann['image_id'] not in image_ids:
                invalid_refs += 1
            if ann['category_id'] not in category_ids:
                invalid_refs += 1
        
        if invalid_refs > 0:
            print(f"  ⚠️ Found {invalid_refs} invalid references")
        else:
            print(f"  ✅ All references valid")
        
        # Check required fields in annotations
        required_ann_fields = ['id', 'image_id', 'category_id', 'location', 'dimensions']
        missing_fields = 0
        
        for ann in data['annotations']:
            for field in required_ann_fields:
                if field not in ann:
                    missing_fields += 1
        
        if missing_fields > 0:
            print(f"  ⚠️ Found {missing_fields} missing annotation fields")
        else:
            print(f"  ✅ All annotation fields present")
        
        print(f"  📊 Summary: {len(data['images'])} images, "
              f"{len(data['categories'])} categories, {len(data['annotations'])} annotations")
        
        return invalid_refs == 0 and missing_fields == 0


def main():
    parser = argparse.ArgumentParser(description="Analyze and convert pipeline outputs to Cube R-CNN format")
    parser.add_argument("--output_dir", required=True, help="Directory containing pipeline outputs")
    parser.add_argument("--analysis_report", default="analysis_report.json", 
                       help="Output file for analysis report")
    parser.add_argument("--convert", action="store_true", 
                       help="Also convert to OMNI3D format")
    parser.add_argument("--omni3d_json", default="converted_omni3d.json",
                       help="Output JSON file in OMNI3D format")
    parser.add_argument("--original_images_dir", 
                       help="Directory containing original images (if different from output_dir)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"❌ Output directory does not exist: {args.output_dir}")
        return
    
    # Run analysis
    analyzer = OutputAnalyzer(args.output_dir)
    results = analyzer.run_complete_analysis()
    analyzer.save_analysis_report(args.analysis_report)
    
    # Optional conversion
    if args.convert:
        print(f"\n🔄 === STARTING CONVERSION ===")
        converter = CubeRCNNConverter(args.output_dir, args.original_images_dir)
        data = converter.load_all_data()
        omni3d_data = converter.convert_to_omni3d_format(data, args.omni3d_json)
        converter.validate_conversion(args.omni3d_json)
        
        print(f"\n🎉 Conversion complete! Next steps:")
        print(f"  1. Review the converted JSON: {args.omni3d_json}")
        print(f"  2. Adjust coordinate systems if needed")
        print(f"  3. Add original images to the dataset")
        print(f"  4. Test with Cube R-CNN training pipeline")


if __name__ == "__main__":
    main()