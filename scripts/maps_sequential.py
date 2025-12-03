#!/usr/bin/env python3
"""
Sequential Google Maps Satellite Downloader
============================================
Single-threaded tile download with optional disk-based processing.
"""

import os
import sys
import time
import tempfile
import shutil
from typing import Tuple, Optional, Dict
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from maps_core import (
    calculate_tile_grid, download_single_tile, stitch_mosaic, stitch_mosaic_streaming
)


def download_satellite_map_sequential(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    zoom: int = 19,
    tile_size_px: int = 640,
    scale: int = 2,
    crop_bottom: int = 40,
    api_key: str = None,
    secret: str = None,
    verbose: bool = True,
    output_path: str = None,
    use_disk: bool = None
) -> Tuple[Optional[Image.Image], Optional[Dict]]:
    """
    Download satellite mosaic using sequential (single-threaded) method.
    
    Args:
        output_path: If provided, saves mosaic directly to this path
        use_disk: Force disk-based (True) or memory-based (False) processing
    
    Returns:
        (mosaic_image, metadata) - mosaic_image is None if output_path is set
    """
    if api_key is None:
        api_key = os.environ.get('GOOGLE_MAPS_API_KEY') or os.environ.get('GMAPS_KEY')
    if secret is None:
        secret = os.environ.get('GOOGLE_MAPS_SECRET')
    
    if not api_key:
        print("[Sequential] ERROR: No Google Maps API key found!")
        return None, None
    
    tile_requests, num_rows, num_cols, metadata = calculate_tile_grid(
        lat_min, lat_max, lon_min, lon_max, zoom, tile_size_px
    )
    total_tiles = len(tile_requests)
    
    # Auto-detect: use disk for large datasets
    if use_disk is None:
        use_disk = total_tiles > 200
    
    if verbose:
        print(f"[Sequential] Downloading {total_tiles} tiles ({num_rows}x{num_cols})")
        print(f"[Sequential]   Zoom: {zoom}, Resolution: ~{metadata['meters_per_pixel']:.2f}m/pixel")
        print(f"[Sequential]   Mode: {'disk' if use_disk else 'memory'}")
    
    start_time = time.time()
    temp_dir = None
    
    try:
        if use_disk:
            # Disk-based processing
            temp_dir = tempfile.mkdtemp(prefix='gmaps_seq_')
            tile_files = {}
            
            for idx, req in enumerate(tile_requests):
                img = download_single_tile(
                    req['lat'], req['lon'],
                    zoom, tile_size_px, scale,
                    api_key, secret, crop_bottom
                )
                
                if img:
                    tile_path = os.path.join(temp_dir, f"tile_{req['row']:03d}_{req['col']:03d}.jpg")
                    img.save(tile_path, 'JPEG', quality=92)
                    img.close()
                    tile_files[(req['row'], req['col'])] = tile_path
                
                if verbose and ((idx + 1) % 10 == 0 or idx + 1 == total_tiles):
                    print(f"[Sequential]   Progress: {idx + 1}/{total_tiles} tiles")
            
            success_count = len(tile_files)
            
            if success_count < total_tiles * 0.5:
                print(f"[Sequential] ERROR: Too many failures ({total_tiles - success_count}/{total_tiles})")
                return None, None
            
            if verbose:
                print(f"[Sequential] Downloaded {success_count}/{total_tiles}, stitching...")
            
            if output_path:
                mosaic_path = output_path
            else:
                mosaic_path = os.path.join(temp_dir, 'mosaic.jpg')
            
            mosaic_size = stitch_mosaic_streaming(
                tile_files, num_rows, num_cols,
                tile_size_px, scale, crop_bottom,
                mosaic_path, quality=85
            )
            
            if verbose:
                print(f"[Sequential] Mosaic: {mosaic_size[0]}x{mosaic_size[1]} px")
            
            if output_path:
                mosaic = None
            else:
                mosaic = Image.open(mosaic_path)
        else:
            # In-memory processing
            tiles = []
            for idx, req in enumerate(tile_requests):
                img = download_single_tile(
                    req['lat'], req['lon'],
                    zoom, tile_size_px, scale,
                    api_key, secret, crop_bottom
                )
                tiles.append({
                    'row': req['row'],
                    'col': req['col'],
                    'image': img
                })
                
                if verbose and ((idx + 1) % 10 == 0 or idx + 1 == total_tiles):
                    print(f"[Sequential]   Progress: {idx + 1}/{total_tiles} tiles")
            
            success_count = sum(1 for t in tiles if t['image'] is not None)
            
            if success_count < total_tiles * 0.5:
                print(f"[Sequential] ERROR: Too many failures ({total_tiles - success_count}/{total_tiles})")
                return None, None
            
            if verbose:
                print(f"[Sequential] Downloaded {success_count}/{total_tiles}, stitching...")
            
            mosaic = stitch_mosaic(tiles, num_rows, num_cols, tile_size_px, scale, crop_bottom)
            
            if output_path:
                mosaic.save(output_path, 'JPEG', quality=85, optimize=True)
            
            if verbose:
                print(f"[Sequential] Mosaic: {mosaic.size[0]}x{mosaic.size[1]} px")
        
        elapsed = time.time() - start_time
        
        metadata['download_method'] = 'sequential'
        metadata['tiles_success'] = success_count
        metadata['tiles_total'] = total_tiles
        metadata['time'] = elapsed
        metadata['use_disk'] = use_disk
        
        return mosaic, metadata
        
    finally:
        # Cleanup temp directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
