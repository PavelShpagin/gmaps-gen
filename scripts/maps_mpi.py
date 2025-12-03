#!/usr/bin/env python3
"""
MPI-Parallel Google Maps Satellite Downloader
==============================================
Uses MPI to parallelize tile downloads across multiple processes.
Achieves ~10x speedup on multi-core systems.

Usage:
    mpiexec -n 8 python maps_mpi.py --lat-min 50.447 --lat-max 50.453 ...
    
Or import and use download_satellite_map_mpi() function.
"""

import os
import sys
import time
import pickle
from typing import Tuple, Optional, Dict, List
from PIL import Image

# Add scripts folder to path
sys.path.insert(0, os.path.dirname(__file__))
from maps_core import (
    calculate_tile_grid, download_single_tile, stitch_mosaic
)

# Try to import MPI
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    MPI = None


def download_satellite_map_mpi(
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
    Download satellite mosaic using MPI parallelization.
    
    Each MPI rank downloads a subset of tiles, then rank 0 gathers and stitches.
    Falls back to sequential if MPI not available or single process.
    
    Returns:
        (mosaic_image, metadata) on rank 0, (None, None) on other ranks
    """
    if not HAS_MPI:
        # Fallback to sequential
        from maps_sequential import download_satellite_map_sequential
        return download_satellite_map_sequential(
            lat_min, lat_max, lon_min, lon_max,
            zoom, tile_size_px, scale, crop_bottom,
            api_key, secret, verbose
        )
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size == 1:
        # Single process, use sequential
        from maps_sequential import download_satellite_map_sequential
        return download_satellite_map_sequential(
            lat_min, lat_max, lon_min, lon_max,
            zoom, tile_size_px, scale, crop_bottom,
            api_key, secret, verbose
        )
    
    # Get API credentials from environment if not provided
    if api_key is None:
        api_key = os.environ.get('GOOGLE_MAPS_API_KEY') or os.environ.get('GMAPS_KEY')
    if secret is None:
        secret = os.environ.get('GOOGLE_MAPS_SECRET')
    
    if not api_key:
        if rank == 0:
            print("[MPI] ERROR: No Google Maps API key found!")
        return None, None
    
    # Calculate tile grid (all ranks need this)
    tile_requests, num_rows, num_cols, metadata = calculate_tile_grid(
        lat_min, lat_max, lon_min, lon_max, zoom, tile_size_px
    )
    total_tiles = len(tile_requests)
    
    if rank == 0 and verbose:
        print(f"[MPI] Downloading {total_tiles} tiles with {size} processes")
        print(f"[MPI]   Grid: {num_rows}x{num_cols}, Zoom: {zoom}")
    
    # Distribute tiles across ranks
    tiles_per_rank = total_tiles // size
    remainder = total_tiles % size
    
    start_idx = rank * tiles_per_rank + min(rank, remainder)
    if rank < remainder:
        tiles_per_rank += 1
    end_idx = start_idx + tiles_per_rank
    
    my_tiles = tile_requests[start_idx:end_idx]
    
    if verbose:
        print(f"[MPI] Rank {rank}: downloading tiles {start_idx}-{end_idx-1} ({len(my_tiles)} tiles)")
    
    # Download assigned tiles
    local_results = []
    for req in my_tiles:
        img = download_single_tile(
            req['lat'], req['lon'],
            zoom, tile_size_px, scale,
            api_key, secret, crop_bottom
        )
        local_results.append({
            'row': req['row'],
            'col': req['col'],
            'index': req['index'],
            'image': img
        })
    
    if verbose:
        success = sum(1 for t in local_results if t['image'] is not None)
        print(f"[MPI] Rank {rank}: downloaded {success}/{len(my_tiles)} tiles")
    
    # Gather results to rank 0
    # Serialize images for MPI transfer
    serialized = []
    for tile in local_results:
        if tile['image'] is not None:
            from io import BytesIO
            buf = BytesIO()
            tile['image'].save(buf, format='JPEG', quality=95)
            serialized.append({
                'row': tile['row'],
                'col': tile['col'],
                'index': tile['index'],
                'data': buf.getvalue()
            })
        else:
            serialized.append({
                'row': tile['row'],
                'col': tile['col'],
                'index': tile['index'],
                'data': None
            })
    
    all_serialized = comm.gather(serialized, root=0)
    
    if rank == 0:
        # Reconstruct tiles
        all_tiles = []
        for rank_tiles in all_serialized:
            for tile in rank_tiles:
                if tile['data'] is not None:
                    from io import BytesIO
                    img = Image.open(BytesIO(tile['data']))
                    all_tiles.append({
                        'row': tile['row'],
                        'col': tile['col'],
                        'index': tile['index'],
                        'image': img
                    })
                else:
                    all_tiles.append({
                        'row': tile['row'],
                        'col': tile['col'],
                        'index': tile['index'],
                        'image': None
                    })
        
        # Sort by index to ensure correct order
        all_tiles.sort(key=lambda t: t['index'])
        
        success_count = sum(1 for t in all_tiles if t['image'] is not None)
        if verbose:
            print(f"[MPI] Gathered {success_count}/{total_tiles} tiles, stitching...")
        
        # Stitch mosaic
        mosaic = stitch_mosaic(all_tiles, num_rows, num_cols, tile_size_px, scale, crop_bottom)
        
        metadata['download_method'] = 'mpi'
        metadata['mpi_ranks'] = size
        metadata['tiles_success'] = success_count
        metadata['tiles_total'] = total_tiles
        
        if verbose:
            print(f"[MPI] Mosaic: {mosaic.size[0]}x{mosaic.size[1]} px")
        
        return mosaic, metadata
    else:
        return None, None


def main():
    """Command-line interface for MPI download."""
    import argparse
    
    parser = argparse.ArgumentParser(description='MPI-parallel Google Maps downloader')
    parser.add_argument('--lat-min', type=float, default=50.447)
    parser.add_argument('--lat-max', type=float, default=50.453)
    parser.add_argument('--lon-min', type=float, default=30.52)
    parser.add_argument('--lon-max', type=float, default=30.53)
    parser.add_argument('--zoom', type=int, default=19)
    parser.add_argument('--tile-size', type=int, default=640)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--output', type=str, default='mosaic_mpi.jpg')
    args = parser.parse_args()
    
    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    mosaic, metadata = download_satellite_map_mpi(
        args.lat_min, args.lat_max,
        args.lon_min, args.lon_max,
        zoom=args.zoom,
        tile_size_px=args.tile_size,
        scale=args.scale
    )
    
    if mosaic is not None:
        mosaic.save(args.output, quality=95)
        print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()

