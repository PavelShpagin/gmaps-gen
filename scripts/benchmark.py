#!/usr/bin/env python3
"""
Benchmark: Sequential vs Parallel Download
===========================================
Tests download performance with optimal configurations.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass


# Benchmark configurations
CONFIGS = {
    'small': {
        'name': 'Small (16 tiles, 4x4)',
        'bounds': (50.445, 50.450, 30.515, 30.525),
        'zoom': 19,
    },
    'medium': {
        'name': 'Medium (36 tiles, 6x6)',
        'bounds': (50.443, 50.453, 30.512, 30.530),
        'zoom': 19,
    },
    'large': {
        'name': 'Large (100+ tiles)',
        'bounds': (50.440, 50.458, 30.505, 30.540),
        'zoom': 19,
    },
    'paper': {
        'name': 'Paper-scale (~462 tiles, 21x22)',
        'bounds': (50.435, 50.465, 30.495, 30.555),
        'zoom': 19,
    }
}


def run_sequential(bounds, zoom):
    """Run sequential download benchmark."""
    from maps_sequential import download_satellite_map_sequential
    
    print("[Sequential] Starting...")
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


def run_fast(bounds, zoom, workers):
    """Run fast parallel download benchmark."""
    from maps_fast import download_satellite_map_fast
    
    print(f"[Fast Parallel] Workers={workers}...")
    start = time.time()
    mosaic, meta = download_satellite_map_fast(
        bounds[0], bounds[1], bounds[2], bounds[3],
        zoom=zoom, max_workers=workers, verbose=True
    )
    elapsed = time.time() - start
    
    if mosaic:
        tiles = meta.get('total_tiles', 0)
        return {
            'method': f'fast_w{workers}',
            'workers': workers,
            'tiles': tiles,
            'success': meta.get('tiles_success', tiles),
            'time': elapsed,
            'throughput': tiles / elapsed if elapsed > 0 else 0
        }
    return None


def main():
    parser = argparse.ArgumentParser(description='Benchmark download methods')
    parser.add_argument('--config', choices=list(CONFIGS.keys()), default='medium')
    parser.add_argument('--workers', nargs='+', type=int, default=[20, 30])
    parser.add_argument('--skip-sequential', action='store_true')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    cfg = CONFIGS[args.config]
    bounds = cfg['bounds']
    zoom = cfg['zoom']
    
    print("=" * 70)
    print(f"BENCHMARK: {cfg['name']}")
    print(f"Zoom: {zoom}")
    print("=" * 70)
    
    results = []
    seq_time = None
    
    # Sequential baseline
    if not args.skip_sequential:
        print()
        result = run_sequential(bounds, zoom)
        if result:
            results.append(result)
            seq_time = result['time']
            print(f"  -> {result['tiles']} tiles in {result['time']:.2f}s ({result['throughput']:.2f} t/s)")
    
    # Fast parallel tests
    for w in args.workers:
        print()
        result = run_fast(bounds, zoom, w)
        if result:
            results.append(result)
            speedup = seq_time / result['time'] if seq_time else 1.0
            print(f"  -> {result['tiles']} tiles in {result['time']:.2f}s ({result['throughput']:.2f} t/s, speedup: {speedup:.1f}x)")
    
    # Summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    best_speedup = 1.0
    for r in results:
        method = r['method']
        if seq_time and r['method'] != 'sequential':
            speedup = seq_time / r['time']
            best_speedup = max(best_speedup, speedup)
            print(f"{method:25}: {r['time']:>7.2f}s, {r['throughput']:>5.1f} t/s, speedup: {speedup:.1f}x")
        else:
            print(f"{method:25}: {r['time']:>7.2f}s, {r['throughput']:>5.1f} t/s")
    
    # Save results
    output_file = args.output or f"benchmark_{args.config}.json"
    output_path = Path(__file__).parent / output_file
    
    data = {
        'config': args.config,
        'zoom': zoom,
        'results': results,
        'best_speedup': best_speedup,
        'sequential_time': seq_time
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
