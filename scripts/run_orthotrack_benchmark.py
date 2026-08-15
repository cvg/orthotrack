#!/usr/bin/env python3
"""
Run OrthoTrack on the MovingDrone test set sequences.
"""

import argparse
import json
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Benchmark OrthoTrack on MovingDrone test set.")
    parser.add_argument('--data_root', type=str, default='data/MovingDrone', help='Path to MovingDrone dataset root')
    parser.add_argument('--output_dir', type=str, default='outputs/benchmark', help='Output directory for benchmark results')
    args, unknown_args = parser.parse_known_args()

    splits_path = Path(args.data_root) / 'splits.json'
    if not splits_path.exists():
        print(f"[ERROR] Splits file not found at {splits_path}")
        print("Please download it or check your --data_root path.")
        return

    with open(splits_path, 'r') as f:
        splits = json.load(f)

    # The test set is the union of test_inPlace and test_outPlace
    test_seqs = sorted(set(splits.get('test_inPlace', [])) | set(splits.get('test_outPlace', [])))
    
    if not test_seqs:
        print("[ERROR] No test sequences found in splits.json")
        return

    print(f"Found {len(test_seqs)} test sequences to process.")
    
    for i, seq in enumerate(test_seqs):
        print(f"\n[{i+1}/{len(test_seqs)}] === Running tracking on {seq} ===")
        seq_dir = Path(args.data_root) / 'scenes' / seq
        out_dir = Path(args.output_dir) / seq
        
        cmd = [
            'python', 'scripts/run_tracking.py',
            '--sequence_dir', str(seq_dir),
            '--output', str(out_dir)
        ] + unknown_args
        
        print(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Tracking failed on sequence {seq} with exit code {e.returncode}")

if __name__ == "__main__":
    main()
