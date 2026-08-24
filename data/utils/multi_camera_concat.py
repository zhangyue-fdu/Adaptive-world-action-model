#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Camera View Concatenation Utility

Simple utility for concatenating three camera views:
- Head camera: Keep original size
- Left/Right wrist cameras: Resize to half and stack vertically

"""
import cv2
import numpy as np
from typing import Optional, Tuple


def resize_and_concatenate_frames(
    head_img: np.ndarray, 
    left_img: np.ndarray, 
    right_img: np.ndarray
) -> Optional[np.ndarray]:
    """
    Concatenate three camera views in T-shape layout:
    - Top: Head camera (keep original size, e.g., 480x640)
    - Bottom left: Left wrist camera (resize to half, e.g., 240x320)
    - Bottom right: Right wrist camera (resize to half, e.g., 240x320)
    Final output: 720x640 (height x width)
        
    Args:
        head_img: Head camera image (keep original size)
        left_img: Left wrist camera image (resize to half size)  
        right_img: Right wrist camera image (resize to half size)
            
    Returns:
        Concatenated image with T-shape layout
    """
    try:
        # Get original dimensions
        orig_h, orig_w = head_img.shape[:2]
            
        # Resize wrist cameras to half size
        half_h, half_w = orig_h // 2, orig_w // 2
        left_resized = cv2.resize(left_img, (half_w, half_h))
        right_resized = cv2.resize(right_img, (half_w, half_h))
            
        # Concatenate left and right wrist cameras horizontally for bottom row
        bottom_row = np.hstack([left_resized, right_resized])
            
        # Create final T-shape layout:
        # Top row: head camera (orig_h x orig_w)
        # Bottom row: combined wrist cameras (half_h x orig_w)
        combined = np.vstack([head_img, bottom_row])
            
        return combined
    except Exception as e:
        return None


def get_concatenated_dimensions(original_shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    Calculate output dimensions for concatenated frame.
    
    Args:
        original_shape: (height, width) of original images
        
    Returns:
        (height, width) of concatenated result
    """
    h, w = original_shape
    # Final: (3w/2) × h
    return h, int(w * 1.5)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Concatenate head/left/right camera views into a T-shape frame.")
    parser.add_argument("--head_image", type=str, default=None, help="Path to head camera image (png/jpg)")
    parser.add_argument("--left_image", type=str, default=None, help="Path to left wrist camera image (png/jpg)")
    parser.add_argument("--right_image", type=str, default=None, help="Path to right wrist camera image (png/jpg)")
    parser.add_argument("--output", type=str, default=None, help="Output image path (png/jpg)")
    args = parser.parse_args()

    if args.head_image and args.left_image and args.right_image and args.output:
        head_img = cv2.imread(args.head_image, cv2.IMREAD_COLOR)
        left_img = cv2.imread(args.left_image, cv2.IMREAD_COLOR)
        right_img = cv2.imread(args.right_image, cv2.IMREAD_COLOR)

        if head_img is None:
            raise FileNotFoundError(f"Failed to read --head_image: {args.head_image}")
        if left_img is None:
            raise FileNotFoundError(f"Failed to read --left_image: {args.left_image}")
        if right_img is None:
            raise FileNotFoundError(f"Failed to read --right_image: {args.right_image}")

        result = resize_and_concatenate_frames(head_img, left_img, right_img)
        if result is None:
            raise RuntimeError("Concatenation failed (resize_and_concatenate_frames returned None).")

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        ok = cv2.imwrite(args.output, result)
        if not ok:
            raise RuntimeError(f"Failed to write output: {args.output}")
        print(f"Saved concatenated image to {args.output} (shape={result.shape})")
    else:
        # Fallback: self-test with dummy images (kept for quick sanity check).
        h, w = 240, 320
        head_img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        left_img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        right_img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        result = resize_and_concatenate_frames(head_img, left_img, right_img)
        if result is not None:
            print(f"Original shape: {head_img.shape}")
            print(f"Concatenated shape: {result.shape}")
            print(f"Expected shape: {get_concatenated_dimensions((h, w))}")
        else:
            print("Concatenation failed")
