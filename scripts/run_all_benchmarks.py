"""
Run all benchmarks - optimized configurations only.
"""
import subprocess
import sys
import json
import os
from pathlib import Path

def run_benchmark(config, workers, skip_sequential=False):
    """Run a single benchmark configuration."""
    cmd = [sys.executable, "benchmark.py", "--config", config, "--workers"] + [str(w) for w in workers]
    if skip_sequential:
        cmd.append("--skip-sequential")
    
    print(f"\n{'='*70}")
    print(f"CONFIG: {config.upper()}, Workers: {workers}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

def main():
    os.chdir(Path(__file__).parent)
    
    print("=" * 70)
    print("RUNNING ALL BENCHMARKS (OPTIMIZED)")
    print("=" * 70)
    
    # Only 2-3 optimal worker configurations
    benchmarks = [
        # Medium: with sequential baseline for speedup calculation
        ("medium", [20, 30], False),
        # Large: skip sequential (too slow), test optimal workers
        ("large", [25, 30], True),
    ]
    
    for config, workers, skip_seq in benchmarks:
        run_benchmark(config, workers, skip_seq)
        print(f"\n[DONE] {config}\n")
    
    print("\n" + "=" * 70)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 70)
    
    # Summary
    for config, _, _ in benchmarks:
        json_file = f"benchmark_{config}.json"
        if os.path.exists(json_file):
            with open(json_file) as f:
                data = json.load(f)
            print(f"\n{config.upper()}:")
            for r in data['results']:
                speedup = data.get('sequential_time', r['time']) / r['time'] if data.get('sequential_time') else 1.0
                print(f"  {r['method']}: {r['time']:.1f}s, {r['throughput']:.1f} t/s, ~{speedup:.1f}x")

if __name__ == "__main__":
    main()
