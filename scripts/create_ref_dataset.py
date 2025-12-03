#!/usr/bin/env python3
"""
Reference Dataset Creator
=========================
Creates a reference tile dataset by cropping from a pre-downloaded mosaic.
Avoids redundant API requests by sampling from existing large satellite image.

Based on NaviLoc paper defaults:
- Reference tile size: 500x500 pixels
- Grid spacing: 40m (configurable: 20m, 10m)
- Zoom level: 19 (~0.19m/pixel at mid-latitudes)

Usage:
    python create_ref_dataset.py --mosaic demo_mosaic.jpg --output refs/
    python create_ref_dataset.py --mosaic demo_mosaic.jpg --spacing 20 --tile-size 500
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from PIL import Image
import csv

sys.path.insert(0, os.path.dirname(__file__))
from maps_core import pixel_to_latlon, latlon_to_pixel


def create_reference_grid(
    mosaic_path: str,
    output_dir: str,
    tile_size_px: int = 500,
    spacing_m: float = 40.0,
    zoom: int = 19,
    center_lat: float = None,
    center_lon: float = None,
    overlap: float = 0.0
) -> Tuple[List[Dict], Dict]:
    """
    Create reference dataset by cropping tiles from a mosaic.
    
    Args:
        mosaic_path: Path to the pre-downloaded mosaic image
        output_dir: Directory to save reference tiles and metadata
        tile_size_px: Size of each reference tile in pixels (default 500)
        spacing_m: Grid spacing in meters (default 40m as in NaviLoc paper)
        zoom: Zoom level used for the mosaic (for coordinate conversion)
        center_lat, center_lon: Center coordinates of the mosaic
        overlap: Tile overlap ratio (0.0 = no overlap, 0.5 = 50% overlap)
    
    Returns:
        (references_list, metadata)
    """
    # Load mosaic
    mosaic = Image.open(mosaic_path)
    mosaic_w, mosaic_h = mosaic.size
    
    # Calculate meters per pixel at this zoom level
    # At zoom 19, ~0.19m/pixel at 50 degrees latitude
    if center_lat is None:
        center_lat = 50.45  # Default (Kyiv area)
    if center_lon is None:
        center_lon = 30.52
    
    meters_per_pixel = 156543.03392 * math.cos(math.radians(center_lat)) / (2 ** zoom)
    
    # Convert spacing to pixels
    spacing_px = int(spacing_m / meters_per_pixel)
    
    # Account for overlap
    step_px = int(spacing_px * (1 - overlap))
    if step_px < 1:
        step_px = 1
    
    # Calculate grid dimensions
    # Ensure we have margin for full tiles
    margin = tile_size_px // 2
    usable_w = mosaic_w - tile_size_px
    usable_h = mosaic_h - tile_size_px
    
    if usable_w <= 0 or usable_h <= 0:
        print(f"ERROR: Mosaic too small ({mosaic_w}x{mosaic_h}) for tile size {tile_size_px}")
        return [], {}
    
    num_cols = max(1, usable_w // step_px + 1)
    num_rows = max(1, usable_h // step_px + 1)
    
    print(f"Creating reference grid:")
    print(f"  Mosaic: {mosaic_w}x{mosaic_h} px")
    print(f"  Tile size: {tile_size_px}x{tile_size_px} px")
    print(f"  Spacing: {spacing_m}m ({spacing_px}px), Step: {step_px}px")
    print(f"  Grid: {num_rows}x{num_cols} = {num_rows * num_cols} tiles")
    print(f"  Resolution: ~{meters_per_pixel:.3f} m/pixel")
    
    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    img_dir = out_path / 'reference_images'
    img_dir.mkdir(exist_ok=True)
    
    # Calculate mosaic bounds in world coordinates
    # Assuming mosaic center is at (center_lat, center_lon)
    cx_px, cy_px = latlon_to_pixel(center_lat, center_lon, zoom)
    mosaic_x_min = cx_px - mosaic_w / 2
    mosaic_y_min = cy_px - mosaic_h / 2
    
    references = []
    ref_idx = 0
    
    for row in range(num_rows):
        for col in range(num_cols):
            # Tile center in mosaic pixel coordinates
            tile_cx = margin + col * step_px + tile_size_px // 2
            tile_cy = margin + row * step_px + tile_size_px // 2
            
            # Crop bounds
            x1 = tile_cx - tile_size_px // 2
            y1 = tile_cy - tile_size_px // 2
            x2 = x1 + tile_size_px
            y2 = y1 + tile_size_px
            
            # Bounds check
            if x2 > mosaic_w or y2 > mosaic_h:
                continue
            
            # Crop tile
            tile = mosaic.crop((x1, y1, x2, y2))
            
            # Convert tile center to lat/lon
            world_x = mosaic_x_min + tile_cx
            world_y = mosaic_y_min + tile_cy
            lat, lon = pixel_to_latlon(world_x, world_y, zoom)
            
            # Save tile
            tile_name = f'ref_{ref_idx:06d}.jpg'
            tile_path = img_dir / tile_name
            tile.save(tile_path, quality=95)
            
            # UTM-like local coordinates (meters from mosaic center)
            local_x = (tile_cx - mosaic_w / 2) * meters_per_pixel
            local_y = (mosaic_h / 2 - tile_cy) * meters_per_pixel  # Y inverted
            
            references.append({
                'name': tile_name,
                'latitude': lat,
                'longitude': lon,
                'x': local_x,
                'y': local_y,
                'row': row,
                'col': col
            })
            ref_idx += 1
    
    # Save CSV
    csv_path = out_path / 'reference.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'latitude', 'longitude', 'x', 'y'])
        writer.writeheader()
        for ref in references:
            writer.writerow({k: ref[k] for k in ['name', 'latitude', 'longitude', 'x', 'y']})
    
    # Save metadata
    metadata = {
        'mosaic_path': str(mosaic_path),
        'mosaic_size': [mosaic_w, mosaic_h],
        'tile_size_px': tile_size_px,
        'spacing_m': spacing_m,
        'spacing_px': spacing_px,
        'step_px': step_px,
        'overlap': overlap,
        'zoom': zoom,
        'meters_per_pixel': meters_per_pixel,
        'center_lat': center_lat,
        'center_lon': center_lon,
        'grid_rows': num_rows,
        'grid_cols': num_cols,
        'total_refs': len(references)
    }
    
    meta_path = out_path / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved {len(references)} reference tiles to {out_path}")
    print(f"  Images: {img_dir}")
    print(f"  CSV: {csv_path}")
    print(f"  Metadata: {meta_path}")
    
    return references, metadata


def main():
    parser = argparse.ArgumentParser(
        description='Create reference dataset from mosaic (NaviLoc-style)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--mosaic', type=str, required=True,
                        help='Path to the mosaic image')
    parser.add_argument('--output', type=str, default='refs',
                        help='Output directory for reference dataset')
    parser.add_argument('--tile-size', type=int, default=500,
                        help='Reference tile size in pixels (NaviLoc default: 500)')
    parser.add_argument('--spacing', type=float, default=40.0,
                        help='Grid spacing in meters (NaviLoc default: 40m)')
    parser.add_argument('--zoom', type=int, default=19,
                        help='Zoom level used for the mosaic')
    parser.add_argument('--center-lat', type=float, default=50.45,
                        help='Center latitude of mosaic')
    parser.add_argument('--center-lon', type=float, default=30.525,
                        help='Center longitude of mosaic')
    parser.add_argument('--overlap', type=float, default=0.0,
                        help='Tile overlap ratio (0.0-0.5)')
    
    args = parser.parse_args()
    
    refs, meta = create_reference_grid(
        mosaic_path=args.mosaic,
        output_dir=args.output,
        tile_size_px=args.tile_size,
        spacing_m=args.spacing,
        zoom=args.zoom,
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        overlap=args.overlap
    )
    
    return 0 if refs else 1


if __name__ == '__main__':
    sys.exit(main())

