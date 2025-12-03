#!/usr/bin/env python3
"""
Sequential Google Maps Satellite Downloader
============================================
Single-threaded tile download implementation.
Used as fallback when MPI is not available.
"""

import os
import sys
from typing import Tuple, Optional, Dict
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from maps_core import (
    calculate_tile_grid, download_single_tile, stitch_mosaic
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
    verbose: bool = True
) -> Tuple[Optional[Image.Image], Optional[Dict]]:
    """
    Download satellite mosaic using sequential (single-threaded) method.
    
    Returns:
        (mosaic_image, metadata) or (None, None) if failed
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
    
    if verbose:
        print(f"[Sequential] Downloading {total_tiles} tiles ({num_rows}x{num_cols})")
        print(f"[Sequential]   Zoom: {zoom}, Resolution: ~{metadata['meters_per_pixel']:.2f}m/pixel")
    
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
    
    metadata['download_method'] = 'sequential'
    metadata['tiles_success'] = success_count
    metadata['tiles_total'] = total_tiles
    
    if verbose:
        print(f"[Sequential] Mosaic: {mosaic.size[0]}x{mosaic.size[1]} px")
    
    return mosaic, metadata

