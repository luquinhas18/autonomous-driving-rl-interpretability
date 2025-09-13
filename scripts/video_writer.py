#!/usr/bin/env python3
"""
Video Writer Script for HighwayEnv Captum IG Visualizations

This script reads attribution and RGB overlay images from the highway_cnn_simple_captum_ig directory
and creates videos from them. It supports creating separate videos for each type or combined videos.
"""

import os
import cv2
import numpy as np
import glob
import re
from pathlib import Path
import argparse
import subprocess
import tempfile


def extract_step_number(filename):
    """Extract step number from filename like 'attribution_step_10_action_3.png'"""
    match = re.search(r'step_(\d+)_', filename)
    return int(match.group(1)) if match else 0


def get_sorted_images(directory, pattern):
    """Get images sorted by step number"""
    image_paths = glob.glob(os.path.join(directory, pattern))
    # Sort by step number
    image_paths.sort(key=lambda x: extract_step_number(os.path.basename(x)))
    return image_paths


def create_video_from_images(image_paths, output_path, fps=10, frame_duration_ms=100):
    """
    Create a video from a list of image paths
    
    Args:
        image_paths: List of paths to images
        output_path: Output video file path
        fps: Frames per second for the video
        frame_duration_ms: Duration to show each frame in milliseconds
    """
    if not image_paths:
        print(f"No images found for pattern")
        return False
    
    # Read the first image to get dimensions
    first_image = cv2.imread(image_paths[0])
    if first_image is None:
        print(f"Could not read first image: {image_paths[0]}")
        return False
    
    height, width, channels = first_image.shape
    
    out = None
    used_codec = None
    
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
   
    
    print(f"Creating video with {len(image_paths)} frames...")
    print(f"Video dimensions: {width}x{height}")
    print(f"FPS: {fps}")
    
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        
        # Resize image if dimensions don't match
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
        
        # Write the frame
        out.write(img)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(image_paths)} frames")
    
    # Release everything
    out.release()
    print(f"Video saved to: {output_path}")
    return True



def create_side_by_side_video(attribution_paths, rgb_paths, output_path, fps=10):
    """
    Create a side-by-side video showing attribution and RGB overlay images
    
    Args:
        attribution_paths: List of attribution image paths
        rgb_paths: List of RGB overlay image paths
        output_path: Output video file path
        fps: Frames per second for the video
    """
    if not attribution_paths or not rgb_paths:
        print("Need both attribution and RGB overlay images for side-by-side video")
        return False
    
    # Read first images to get dimensions
    first_attr = cv2.imread(attribution_paths[0])
    first_rgb = cv2.imread(rgb_paths[0])
    
    if first_attr is None or first_rgb is None:
        print("Could not read first images")
        return False
    
    # Resize images to same height, then combine side by side
    target_height = min(first_attr.shape[0], first_rgb.shape[0])
    target_width = first_attr.shape[1] + first_rgb.shape[1]
    
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    print(f"Creating side-by-side video with {len(attribution_paths)} frames...")
    print(f"Video dimensions: {target_width}x{target_height}")
    print(f"FPS: {fps}")
    
    for i, (attr_path, rgb_path) in enumerate(zip(attribution_paths, rgb_paths)):
        attr_img = cv2.imread(attr_path)
        rgb_img = cv2.imread(rgb_path)
        
        if attr_img is None or rgb_img is None:
            print(f"Warning: Could not read images at step {i}")
            continue
        
        # Resize both images to same height
        attr_resized = cv2.resize(attr_img, (attr_img.shape[1], target_height), interpolation = cv2.INTER_AREA)
        rgb_resized = cv2.resize(rgb_img, (rgb_img.shape[1], target_height), interpolation = cv2.INTER_AREA)
        
        # Combine side by side
        combined = np.hstack([attr_resized, rgb_resized])
        
        out.write(combined)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(attribution_paths)} frames")
    
    out.release()
    print(f"Side-by-side video saved to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Create videos from HighwayEnv Captum IG images')
    parser.add_argument('--input_dir', default='./highway_cnn_simple_captum_ig', 
                       help='Input directory containing images')
    parser.add_argument('--output_dir', default='./videos', 
                       help='Output directory for videos')
    parser.add_argument('--fps', type=int, default=10, 
                       help='Frames per second for output video')
    parser.add_argument('--type', choices=['attribution', 'rgb', 'both', 'side_by_side'], 
                       default='both', help='Type of video to create')
    parser.add_argument('--format', choices=['mp4', 'avi'], 
                       default='mp4', help='Output video format')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get image paths
    attribution_paths = get_sorted_images(args.input_dir, 'attribution_step_*.png')
    rgb_paths = get_sorted_images(args.input_dir, 'rgb_overlay_step_*.png')
    
    video_prefix = "intersection"
    print(f"Found {len(attribution_paths)} attribution images")
    print(f"Found {len(rgb_paths)} RGB overlay images")
    
    if args.type in ['attribution', 'both']:
        if attribution_paths:
            output_path = os.path.join(args.output_dir, f'{video_prefix}_attribution_video.{args.format}')
            create_video_from_images(attribution_paths, output_path, args.fps)
        else:
            print("No attribution images found")
    
    if args.type in ['rgb', 'both']:
        if rgb_paths:
            output_path = os.path.join(args.output_dir, f'{video_prefix}_rgb_overlay_video.{args.format}')
            create_video_from_images(rgb_paths, output_path, args.fps)
        else:
            print("No RGB overlay images found")
    
    if args.type == 'side_by_side':
        if attribution_paths and rgb_paths:
            output_path = os.path.join(args.output_dir, f'{video_prefix}_side_by_side_video.{args.format}')
            create_side_by_side_video(attribution_paths, rgb_paths, output_path, args.fps)
        else:
            print("Need both attribution and RGB overlay images for side-by-side video")


if __name__ == "__main__":
    main()
