#!/usr/bin/env python3
"""
Google Maps Reference Dataset Generator
========================================
Generates satellite imagery datasets for visual localization.

Usage:
    python generate_dataset.py --lat 50.45 --lon 30.52 --width 800 --height 800
    python generate_dataset.py --lat 50.45 --lon 30.52 --left 400 --right 400 --up 400 --down 400

Output:
    output/<method>/
        map.jpg          - Compressed full mosaic
        ref/             - Reference tile database
            tile_00000.jpg
            tile_00001.jpg
            ...
        ref.csv          - Tile metadata (id, filename, x, y, lat, lon, etc.)
        results.json     - Benchmark results
"""

import os
import sys
import csv
import time
import json
import math
import argparse
from pathlib import Path
from PIL import Image

# Handle large images
Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass

from maps_core import latlon_to_pixel, pixel_to_latlon, calculate_tile_grid
from maps_fast import download_satellite_map_fast
from maps_sequential import download_satellite_map_sequential

OUTPUT_DIR = Path(__file__).parent.parent / 'output'


def meters_to_degrees(meters: float, lat: float, direction: str = 'lat') -> float:
    """Convert meters to degrees at given latitude."""
    if direction == 'lat':
        return meters / 111000.0
    else:  # lon
        return meters / (111000.0 * math.cos(math.radians(lat)))


def calculate_bounds(
    center_lat: float,
    center_lon: float,
    left_m: float,
    right_m: float,
    up_m: float,
    down_m: float
) -> tuple:
    """Calculate bounding box from center point and distances in meters."""
    lat_up = meters_to_degrees(up_m, center_lat, 'lat')
    lat_down = meters_to_degrees(down_m, center_lat, 'lat')
    lon_left = meters_to_degrees(left_m, center_lat, 'lon')
    lon_right = meters_to_degrees(right_m, center_lat, 'lon')
    
    lat_min = center_lat - lat_down
    lat_max = center_lat + lat_up
    lon_min = center_lon - lon_left
    lon_max = center_lon + lon_right
    
    return (lat_min, lat_max, lon_min, lon_max)


def create_reference_database(
    mosaic_path: str,
    output_dir: str,
    tile_size: int,
    spacing: int,
    bounds: tuple,
    zoom: int,
    verbose: bool = True
) -> list:
    """
    Create reference tile database from mosaic.
    
    Args:
        mosaic_path: Path to source mosaic
        output_dir: Output directory for tiles
        tile_size: Tile size in pixels
        spacing: Spacing between tile centers in pixels
        bounds: (lat_min, lat_max, lon_min, lon_max)
        zoom: Zoom level used
    
    Returns:
        List of tile metadata dicts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    img = Image.open(mosaic_path)
    width, height = img.size
    
    lat_min, lat_max, lon_min, lon_max = bounds
    
    # Calculate pixel to geo mapping
    px_per_lon = width / (lon_max - lon_min)
    px_per_lat = height / (lat_max - lat_min)
    
    tiles = []
    tile_idx = 0
    half = tile_size // 2
    
    # Generate tiles with specified spacing
    for y in range(half, height - half, spacing):
        for x in range(half, width - half, spacing):
            # Extract tile
            x1, y1 = x - half, y - half
            x2, y2 = x + half, y + half
            
            tile = img.crop((x1, y1, x2, y2))
            
            # Calculate geo coordinates of tile center
            tile_lon = lon_min + x / px_per_lon
            tile_lat = lat_max - y / px_per_lat  # Y is inverted
            
            # Save tile
            filename = f"tile_{tile_idx:05d}.jpg"
            filepath = os.path.join(output_dir, filename)
            tile.save(filepath, 'JPEG', quality=90)
            tile.close()
            
            tiles.append({
                'id': tile_idx,
                'filename': filename,
                'x_px': x,
                'y_px': y,
                'x1_px': x1,
                'y1_px': y1,
                'lat': tile_lat,
                'lon': tile_lon,
                'size_px': tile_size
            })
            
            tile_idx += 1
    
    img.close()
    
    if verbose:
        print(f"[RefDB] Generated {len(tiles)} reference tiles")
    
    return tiles


def save_ref_csv(tiles: list, output_path: str):
    """Save tile metadata to CSV."""
    if not tiles:
        return
    
    fieldnames = ['id', 'filename', 'x_px', 'y_px', 'x1_px', 'y1_px', 'lat', 'lon', 'size_px']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tiles)


def run_pipeline(
    name: str,
    bounds: tuple,
    zoom: int,
    tile_size: int,
    spacing: int,
    workers: int = None,
    verbose: bool = True
) -> dict:
    """
    Run complete dataset generation pipeline.
    
    Args:
        name: Output folder name ('parallel' or 'sequential')
        bounds: (lat_min, lat_max, lon_min, lon_max)
        zoom: Google Maps zoom level
        tile_size: Reference tile size in pixels
        spacing: Spacing between tile centers in pixels
        workers: Number of parallel workers (None for sequential)
    
    Returns:
        Results dictionary
    """
    method_dir = OUTPUT_DIR / name
    method_dir.mkdir(parents=True, exist_ok=True)
    
    map_path = method_dir / 'map.jpg'
    ref_dir = method_dir / 'ref'
    csv_path = method_dir / 'ref.csv'
    
    print(f"\n{'='*70}")
    print(f"{name.upper()}" + (f" ({workers} workers)" if workers else " (sequential)"))
    print('='*70)
    
    # Download tiles and stitch mosaic
    start = time.time()
    
    if workers:
        mosaic, meta = download_satellite_map_fast(
            bounds[0], bounds[1], bounds[2], bounds[3],
            zoom=zoom,
            max_workers=workers,
            verbose=verbose,
            use_disk=True
        )
    else:
        mosaic, meta = download_satellite_map_sequential(
            bounds[0], bounds[1], bounds[2], bounds[3],
            zoom=zoom,
            verbose=verbose,
            use_disk=True
        )
    
    download_time = time.time() - start
    
    if not mosaic or not meta:
        print(f"[{name}] FAILED - no mosaic generated")
        return None
    
    # Save high quality map
    print(f"[{name}] Saving map.jpg...")
    mosaic.save(str(map_path), 'JPEG', quality=90)
    mosaic_size = mosaic.size
    
    # Save compressed map for visualization (max 4000px, lower quality)
    compressed_path = method_dir / 'compressed_map.jpg'
    print(f"[{name}] Saving compressed_map.jpg...")
    max_dim = 4000
    if mosaic_size[0] > max_dim or mosaic_size[1] > max_dim:
        ratio = min(max_dim / mosaic_size[0], max_dim / mosaic_size[1])
        new_size = (int(mosaic_size[0] * ratio), int(mosaic_size[1] * ratio))
        compressed = mosaic.resize(new_size, Image.LANCZOS)
        compressed.save(str(compressed_path), 'JPEG', quality=75)
        compressed.close()
    else:
        mosaic.save(str(compressed_path), 'JPEG', quality=75)
    
    mosaic.close()
    
    # Generate reference database
    print(f"[{name}] Generating reference tiles...")
    tiles = create_reference_database(
        str(map_path),
        str(ref_dir),
        tile_size=tile_size,
        spacing=spacing,
        bounds=bounds,
        zoom=zoom,
        verbose=verbose
    )
    
    # Save CSV
    print(f"[{name}] Saving ref.csv...")
    save_ref_csv(tiles, str(csv_path))
    
    total_time = time.time() - start
    throughput = meta['tiles_total'] / download_time if download_time > 0 else 0
    
    result = {
        'method': name,
        'workers': workers,
        'api_tiles': meta['tiles_total'],
        'grid': f"{meta['num_rows']}x{meta['num_cols']}",
        'download_time': download_time,
        'total_time': total_time,
        'throughput': throughput,
        'mosaic_size': list(mosaic_size),
        'ref_tiles': len(tiles),
        'tile_size': tile_size,
        'spacing': spacing
    }
    
    # Save results JSON
    with open(method_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"[{name}] Done: {download_time:.1f}s download, {throughput:.1f} t/s, {len(tiles)} refs")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Generate satellite reference dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Location
    parser.add_argument('--lat', type=float, default=50.450, help='Center latitude')
    parser.add_argument('--lon', type=float, default=30.525, help='Center longitude')
    
    # Coverage (symmetric)
    parser.add_argument('--width', type=float, help='Total width in meters (overrides left/right)')
    parser.add_argument('--height', type=float, help='Total height in meters (overrides up/down)')
    
    # Coverage (asymmetric)
    parser.add_argument('--left', type=float, default=400, help='Left distance in meters')
    parser.add_argument('--right', type=float, default=400, help='Right distance in meters')
    parser.add_argument('--up', type=float, default=400, help='Up distance in meters')
    parser.add_argument('--down', type=float, default=400, help='Down distance in meters')
    
    # Tile parameters
    parser.add_argument('--tile-size', type=int, default=256, help='Reference tile size in pixels')
    parser.add_argument('--spacing', type=int, default=256, help='Spacing between tile centers')
    parser.add_argument('--zoom', type=int, default=19, help='Google Maps zoom level')
    
    # Processing
    parser.add_argument('--workers', type=int, default=60, help='Parallel workers')
    parser.add_argument('--parallel-only', action='store_true', help='Skip sequential')
    parser.add_argument('--sequential-only', action='store_true', help='Skip parallel')
    
    args = parser.parse_args()
    
    # Handle symmetric width/height
    left = args.width / 2 if args.width else args.left
    right = args.width / 2 if args.width else args.right
    up = args.height / 2 if args.height else args.up
    down = args.height / 2 if args.height else args.down
    
    # Calculate bounds
    bounds = calculate_bounds(args.lat, args.lon, left, right, up, down)
    
    coverage_width = left + right
    coverage_height = up + down
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("REFERENCE DATASET GENERATOR")
    print("=" * 70)
    print(f"Center:    {args.lat:.6f}, {args.lon:.6f}")
    print(f"Coverage:  {coverage_width:.0f}m x {coverage_height:.0f}m")
    print(f"Bounds:    {bounds[0]:.6f} to {bounds[1]:.6f} lat")
    print(f"           {bounds[2]:.6f} to {bounds[3]:.6f} lon")
    print(f"Tile size: {args.tile_size}px, Spacing: {args.spacing}px")
    print(f"Zoom:      {args.zoom}")
    
    results = {}
    
    # Parallel
    if not args.sequential_only:
        results['parallel'] = run_pipeline(
            'parallel', bounds, args.zoom,
            args.tile_size, args.spacing,
            workers=args.workers
        )
    
    # Sequential
    if not args.parallel_only:
        results['sequential'] = run_pipeline(
            'sequential', bounds, args.zoom,
            args.tile_size, args.spacing,
            workers=None
        )
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    for name, r in results.items():
        if r:
            w = f"({r['workers']}w)" if r['workers'] else "(seq)"
            print(f"{name:12} {w:8}: {r['download_time']:>6.1f}s, {r['throughput']:>5.1f} t/s, {r['ref_tiles']} refs")
    
    if 'parallel' in results and 'sequential' in results:
        if results['parallel'] and results['sequential']:
            speedup = results['sequential']['download_time'] / results['parallel']['download_time']
            print(f"\nSPEEDUP: {speedup:.1f}x")
            results['speedup'] = speedup
    
    # Save combined results
    with open(OUTPUT_DIR / 'benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("OUTPUT FILES")
    print("=" * 70)
    for name in ['parallel', 'sequential']:
        if name in results and results[name]:
            d = OUTPUT_DIR / name
            print(f"{d}/")
            print(f"  map.jpg            - Full quality mosaic")
            print(f"  compressed_map.jpg - Small version for visualization")
            print(f"  ref/*.jpg          - {results[name]['ref_tiles']} reference tiles")
            print(f"  ref.csv            - Tile metadata")
            print(f"  results.json       - Benchmark data")


if __name__ == '__main__':
    main()
