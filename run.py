#!/usr/bin/env python3
"""
NaviLoc Demo: Google Maps API Dataset Pipeline
===============================================
Downloads satellite mosaic and creates reference dataset.

Usage:
    # Fast parallel download (default):
    python run.py
    
    # With custom workers:
    python run.py --workers 15
    
    # Create reference dataset from existing mosaic:
    python run.py --refs-only --mosaic demo_mosaic.jpg
    
    # Benchmark mode:
    python run.py --benchmark
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add scripts folder to path
SCRIPT_DIR = Path(__file__).parent / 'scripts'
sys.path.insert(0, str(SCRIPT_DIR))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')
except ImportError:
    pass


def download_mosaic(args):
    """Download satellite mosaic using fast parallel method."""
    from maps_fast import download_satellite_map_fast
    
    api_key = os.environ.get('GOOGLE_MAPS_API_KEY') or os.environ.get('GMAPS_KEY')
    if not api_key:
        print("ERROR: Set GOOGLE_MAPS_API_KEY in .env")
        return None, None, 0
    
    lat_min = float(os.environ.get('LAT_MIN', args.lat_min))
    lat_max = float(os.environ.get('LAT_MAX', args.lat_max))
    lon_min = float(os.environ.get('LON_MIN', args.lon_min))
    lon_max = float(os.environ.get('LON_MAX', args.lon_max))
    
    print(f"Bounds: lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]")
    print(f"Zoom: {args.zoom}, Workers: {args.workers}")
    
    start_time = time.time()
    
    mosaic, metadata = download_satellite_map_fast(
        lat_min, lat_max, lon_min, lon_max,
        zoom=args.zoom,
        tile_size_px=args.tile_size,
        scale=args.scale,
        max_workers=args.workers,
        rate_limit=args.rate,
        verbose=True
    )
    
    elapsed = time.time() - start_time
    return mosaic, metadata, elapsed


def create_refs(args):
    """Create reference dataset from mosaic."""
    from create_ref_dataset import create_reference_grid
    
    center_lat = (args.lat_min + args.lat_max) / 2
    center_lon = (args.lon_min + args.lon_max) / 2
    
    refs, meta = create_reference_grid(
        mosaic_path=args.mosaic,
        output_dir=args.refs_output,
        tile_size_px=args.ref_tile_size,
        spacing_m=args.ref_spacing,
        zoom=args.zoom,
        center_lat=center_lat,
        center_lon=center_lon
    )
    return refs, meta


def run_benchmark(args):
    """Run speed benchmark."""
    from benchmark import run_benchmark as run_bench
    run_bench(args.benchmark_config, args.zoom, [10, 15], args.rate)


def main():
    parser = argparse.ArgumentParser(
        description='NaviLoc Demo: Google Maps Dataset Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Bounds (default: ~700m area for demo)
    parser.add_argument('--lat-min', type=float, default=50.447)
    parser.add_argument('--lat-max', type=float, default=50.453)
    parser.add_argument('--lon-min', type=float, default=30.520)
    parser.add_argument('--lon-max', type=float, default=30.530)
    
    # Download params
    parser.add_argument('--zoom', type=int, default=19)
    parser.add_argument('--tile-size', type=int, default=640)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--output', type=str, default='demo_mosaic.jpg')
    
    # Parallel params
    parser.add_argument('--workers', type=int, default=10,
                        help='Number of parallel download workers')
    parser.add_argument('--rate', type=float, default=25.0,
                        help='Rate limit (requests per second)')
    
    # Reference dataset params (NaviLoc paper defaults)
    parser.add_argument('--ref-tile-size', type=int, default=256,
                        help='Reference tile size (NaviLoc paper: 256px)')
    parser.add_argument('--ref-spacing', type=float, default=40.0,
                        help='Reference grid spacing in meters (NaviLoc: 40m)')
    parser.add_argument('--refs-output', type=str, default='refs',
                        help='Reference dataset output directory')
    
    # Modes
    parser.add_argument('--benchmark', action='store_true',
                        help='Run speed benchmark')
    parser.add_argument('--benchmark-config', type=str, default='medium',
                        choices=['small', 'medium', 'large', 'paper'])
    parser.add_argument('--refs-only', action='store_true',
                        help='Only create refs from existing mosaic')
    parser.add_argument('--mosaic', type=str, default='demo_mosaic.jpg',
                        help='Existing mosaic path (for --refs-only)')
    parser.add_argument('--skip-refs', action='store_true',
                        help='Skip reference dataset creation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NaviLoc Demo: Google Maps API Dataset Pipeline")
    print("=" * 60)
    
    # Benchmark mode
    if args.benchmark:
        run_benchmark(args)
        return 0
    
    # Refs-only mode
    if args.refs_only:
        print(f"\nCreating references from: {args.mosaic}")
        refs, meta = create_refs(args)
        return 0 if refs else 1
    
    # Full pipeline
    print(f"\n[Step 1] Downloading satellite mosaic (fast parallel)...")
    mosaic, metadata, elapsed = download_mosaic(args)
    
    if mosaic is None:
        print("ERROR: Failed to download mosaic")
        return 1
    
    mosaic.save(args.output, quality=95)
    print(f"\nSaved: {args.output} ({mosaic.size[0]}x{mosaic.size[1]} px)")
    print(f"Download time: {elapsed:.2f}s")
    if metadata:
        print(f"Throughput: {metadata.get('throughput', 0):.1f} tiles/sec")
    
    if not args.skip_refs:
        print(f"\n[Step 2] Creating reference dataset...")
        args.mosaic = args.output
        refs, meta = create_refs(args)
        
        if refs:
            print(f"\nCreated {len(refs)} reference tiles")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
