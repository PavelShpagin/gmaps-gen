#!/usr/bin/env python3
"""
Benchmark: Sequential vs Parallel Download
===========================================
Tests moderate (6x6) and paper-scale (21x22) datasets.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass


# Exact configurations
CONFIGS = {
    'moderate': {
        'name': 'Moderate (6x6 = 36 tiles)',
        'bounds': (50.4465, 50.4535, 30.5150, 30.5290),  # ~700m x 900m
        'zoom': 19,
        'expected_tiles': 36,
    },
    'paper': {
        'name': 'Paper-scale (21x22 = 462 tiles)',
        'bounds': (50.440, 50.465, 30.500, 30.560),  # ~800m x ~840m as in paper
        'zoom': 19,
        'expected_tiles': 462,
    }
}


def run_sequential(bounds, zoom):
    """Run sequential download."""
    from maps_sequential import download_satellite_map_sequential
    
    print("\n[Sequential] Starting...")
    start = time.time()
    mosaic, meta = download_satellite_map_sequential(
        bounds[0], bounds[1], bounds[2], bounds[3],
        zoom=zoom, verbose=True
    )
    elapsed = time.time() - start
    
    if mosaic:
        tiles = meta.get('total_tiles', 0)
        return {
            'method': 'sequential',
            'tiles': tiles,
            'success': meta.get('tiles_success', tiles),
            'time': elapsed,
            'throughput': tiles / elapsed if elapsed > 0 else 0
        }
    return None


def run_parallel(bounds, zoom, workers=25):
    """Run parallel download."""
    from maps_fast import download_satellite_map_fast
    
    print(f"\n[Parallel] Workers={workers}...")
    start = time.time()
    mosaic, meta = download_satellite_map_fast(
        bounds[0], bounds[1], bounds[2], bounds[3],
        zoom=zoom, max_workers=workers, verbose=True
    )
    elapsed = time.time() - start
    
    if mosaic:
        tiles = meta.get('total_tiles', 0)
        return {
            'method': f'parallel_w{workers}',
            'workers': workers,
            'tiles': tiles,
            'success': meta.get('tiles_success', tiles),
            'time': elapsed,
            'throughput': tiles / elapsed if elapsed > 0 else 0
        }
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=list(CONFIGS.keys()), required=True)
    parser.add_argument('--workers', type=int, default=25)
    parser.add_argument('--parallel-only', action='store_true')
    parser.add_argument('--sequential-only', action='store_true')
    args = parser.parse_args()
    
    cfg = CONFIGS[args.config]
    bounds = cfg['bounds']
    zoom = cfg['zoom']
    
    print("=" * 70)
    print(f"BENCHMARK: {cfg['name']}")
    print(f"Expected: ~{cfg['expected_tiles']} tiles")
    print("=" * 70)
    
    results = []
    seq_time = None
    
    # Sequential
    if not args.parallel_only:
        result = run_sequential(bounds, zoom)
        if result:
            results.append(result)
            seq_time = result['time']
    
    # Parallel
    if not args.sequential_only:
        result = run_parallel(bounds, zoom, args.workers)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    for r in results:
        if seq_time and 'parallel' in r['method']:
            speedup = seq_time / r['time']
            print(f"{r['method']:20}: {r['time']:>8.2f}s, {r['throughput']:>6.2f} t/s, {r['success']}/{r['tiles']} ok, SPEEDUP: {speedup:.1f}x")
        else:
            print(f"{r['method']:20}: {r['time']:>8.2f}s, {r['throughput']:>6.2f} t/s, {r['success']}/{r['tiles']} ok")
    
    # Save
    output_path = Path(__file__).parent / f"benchmark_{args.config}.json"
    data = {
        'config': args.config,
        'results': results,
        'sequential_time': seq_time
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
