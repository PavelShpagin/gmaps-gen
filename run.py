#!/usr/bin/env python3
"""
Quick Demo - Generate Reference Dataset
=======================================
Simple wrapper for generate_dataset.py with sensible defaults.

Usage:
    python run.py                    # Default 800x800m area
    python run.py --width 1000       # 1km x 1km area
    python run.py --parallel-only    # Skip sequential benchmark
"""

import subprocess
import sys
from pathlib import Path

def main():
    script = Path(__file__).parent / 'scripts' / 'generate_dataset.py'
    
    # Pass all arguments to generate_dataset.py
    args = [sys.executable, str(script)] + sys.argv[1:]
    
    # Default to parallel-only for quick demo
    if len(sys.argv) == 1:
        args.extend(['--width', '800', '--height', '800', '--parallel-only'])
        print("Running quick demo (800x800m, parallel only)...")
        print("For full benchmark: python run.py --width 800 --height 800")
        print()
    
    subprocess.run(args)


if __name__ == '__main__':
    main()
