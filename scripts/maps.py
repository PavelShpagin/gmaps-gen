#!/usr/bin/env python3
"""
Google Maps Satellite Imagery Downloader
========================================
Unified interface with automatic async/MPI detection.

Usage:
    # Default (async - fastest):
    python -c "from maps import download_satellite_map; ..."
    
    # Sequential (fallback):
    python -c "from maps import download_satellite_map; ..." --force-sequential
"""

import os
import sys
from typing import Tuple, Optional, Dict
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

# Check for async support
try:
    import aiohttp
    _HAS_ASYNC = True
except ImportError:
    _HAS_ASYNC = False

# Check for MPI
try:
    from mpi4py import MPI
    _comm = MPI.COMM_WORLD
    _HAS_MPI = _comm.Get_size() > 1
except ImportError:
    _HAS_MPI = False


def download_satellite_map(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    zoom: int = 19,
    tile_size_px: int = 640,
    scale: int = 2,
    crop_bottom: int = 40,
    api_key: str = None,
    secret: str = None,
    verbose: bool = True,
    force_sequential: bool = False,
    max_concurrent: int = 50
) -> Tuple[Optional[Image.Image], Optional[Dict]]:
    """
    Download and stitch Google Maps satellite imagery.
    
    Automatically uses async I/O (fastest) if aiohttp is available.
    Falls back to MPI or sequential.
    
    Args:
        lat_min, lat_max: Latitude bounds
        lon_min, lon_max: Longitude bounds
        zoom: Google Maps zoom level (18-20 recommended)
        tile_size_px: Tile size in pixels (max 640)
        scale: Image scale factor (1 or 2)
        crop_bottom: Pixels to crop from bottom (watermark)
        api_key: Google Maps API key (reads from env if None)
        secret: URL signing secret (reads from env if None)
        verbose: Print progress messages
        force_sequential: Force sequential mode
        max_concurrent: Max concurrent async requests (default 50)
    
    Returns:
        (mosaic_image, metadata) or (None, None) if failed
    """
    if force_sequential:
        from maps_sequential import download_satellite_map_sequential
        return download_satellite_map_sequential(
            lat_min, lat_max, lon_min, lon_max,
            zoom, tile_size_px, scale, crop_bottom,
            api_key, secret, verbose
        )
    
    # Prefer fast parallel (ThreadPool with rate limiting)
    try:
        from maps_fast import download_satellite_map_fast
        return download_satellite_map_fast(
            lat_min, lat_max, lon_min, lon_max,
            zoom, tile_size_px, scale, crop_bottom,
            api_key, secret,
            max_workers=max_concurrent,
            rate_limit=25.0,
            verbose=verbose
        )
    except ImportError:
        pass
    
    # Fallback to MPI if available
    if _HAS_MPI:
        from maps_mpi import download_satellite_map_mpi
        return download_satellite_map_mpi(
            lat_min, lat_max, lon_min, lon_max,
            zoom, tile_size_px, scale, crop_bottom,
            api_key, secret, verbose
        )
    
    # Final fallback: sequential
    from maps_sequential import download_satellite_map_sequential
    return download_satellite_map_sequential(
        lat_min, lat_max, lon_min, lon_max,
        zoom, tile_size_px, scale, crop_bottom,
        api_key, secret, verbose
    )


# Re-export core utilities
from maps_core import (
    latlon_to_pixel,
    pixel_to_latlon,
    calculate_tile_grid,
    download_single_tile,
    stitch_mosaic
)

