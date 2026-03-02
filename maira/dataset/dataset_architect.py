"""
MAIRA — Dataset Architect
Reads parsed result files and intelligently prepares train/test splits.
Writes a justification for every single decision it makes.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scanner'))
from result_parser import ParsedResult, parse_file


@dataclass
class DatasetDecision:
    file_path:     str
    dataset_type:  str          # timeseries / episodic / tabular
    total_records: int
    train_size:    int
    test_size:     int
    val_size:      int
    split_ratio:   str
    justification: List[str] = field(default_factory=list)
    train_data:    Any = None
    test_data:     Any = None
    val_data:      Any = None


def _split_episodic(data: List[Dict], path: str) -> DatasetDecision:
    """Split episode-based JSON data."""
    n       = len(data)
    reasons = []

    # Sort by episode_id if present
    if 'episode_id' in data[0]:
        data = sorted(data, key=lambda x: x['episode_id'])
        reasons.append("Sorted by episode_id to preserve temporal order.")

    # Choose split ratio based on dataset size
    if n < 20:
        train_r, val_r, test_r = 0.7, 0.15, 0.15
        reasons.append(f"Small dataset ({n} episodes) — using 70/15/15 split to preserve test integrity.")
    elif n < 100:
        train_r, val_r, test_r = 0.75, 0.1, 0.15
        reasons.append(f"Medium dataset ({n} episodes) — using 75/10/15 split.")
    else:
        train_r, val_r, test_r = 0.8, 0.1, 0.1
        reasons.append(f"Large dataset ({n} episodes) — using 80/10/10 split.")

    reasons.append("No random shuffle applied — episodic RL data has temporal dependency.")
    reasons.append("Test set uses final episodes — simulates evaluating on unseen future behavior.")

    train_end = int(n * train_r)
    val_end   = int(n * (train_r + val_r))

    train = data[:train_end]
    val   = data[train_end:val_end]
    test  = data[val_end:]

    return DatasetDecision(
        file_path     = path,
        dataset_type  = "episodic",
        total_records = n,
        train_size    = len(train),
        val_size      = len(val),
        test_size     = len(test),
        split_ratio   = f"{int(train_r*100)}/{int(val_r*100)}/{int(test_r*100)}",
        justification = reasons,
        train_data    = train,
        val_data      = val,
        test_data     = test,
    )


def _split_timeseries(arrays: Dict[str, np.ndarray], path: str) -> DatasetDecision:
    """Split timeseries NPZ data (evaluations over training steps)."""
    timesteps = arrays.get('timesteps', np.array([]))
    results   = arrays.get('results',   np.array([]))
    n         = len(timesteps)
    reasons   = []

    reasons.append(f"NPZ contains {n} evaluation checkpoints across training.")
    reasons.append("Timeseries split — no shuffle. Earlier checkpoints = train, later = test.")
    reasons.append("This tests whether MAIRA can predict performance trajectory from early training.")

    train_end = int(n * 0.7)
    val_end   = int(n * 0.85)

    return DatasetDecision(
        file_path     = path,
        dataset_type  = "timeseries",
        total_records = n,
        train_size    = train_end,
        val_size      = val_end - train_end,
        test_size     = n - val_end,
        split_ratio   = "70/15/15",
        justification = reasons,
        train_data    = {"timesteps": timesteps[:train_end],   "results": results[:train_end]},
        val_data      = {"timesteps": timesteps[train_end:val_end], "results": results[train_end:val_end]},
        test_data     = {"timesteps": timesteps[val_end:],     "results": results[val_end:]},
    )


def architect_dataset(result_files: List[str], root: str) -> List[DatasetDecision]:
    """Process all result files and return split decisions."""
    decisions = []

    for rel_path in result_files:
        full = Path(root) / rel_path
        ext  = full.suffix.lower()

        if ext == '.json':
            try:
                with open(full) as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    dec = _split_episodic(data, rel_path)
                    decisions.append(dec)
            except Exception as e:
                pass

        elif ext == '.npz':
            try:
                arrays = np.load(full, allow_pickle=True)
                dec = _split_timeseries(
                    {k: arrays[k] for k in arrays.files}, rel_path
                )
                decisions.append(dec)
            except Exception as e:
                pass

    return decisions


def print_decisions(decisions: List[DatasetDecision]) -> None:
    print("\n" + "="*60)
    print("  MAIRA — Dataset Architect")
    print("="*60)

    for d in decisions:
        print(f"\n  File:         {d.file_path}")
        print(f"  Type:         {d.dataset_type}")
        print(f"  Total:        {d.total_records} records")
        print(f"  Split:        {d.split_ratio}  "
              f"(train={d.train_size} / val={d.val_size} / test={d.test_size})")
        print(f"  Justification:")
        for j in d.justification:
            print(f"    → {j}")

    print()


def save_decisions(decisions: List[DatasetDecision], root: str) -> None:
    """Save split metadata to JSON."""
    out = []
    for d in decisions:
        out.append({
            "file_path":     d.file_path,
            "dataset_type":  d.dataset_type,
            "total_records": d.total_records,
            "train_size":    d.train_size,
            "val_size":      d.val_size,
            "test_size":     d.test_size,
            "split_ratio":   d.split_ratio,
            "justification": d.justification,
        })

    out_path = Path(root) / "maira_dataset_decisions.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Decisions saved to {out_path}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scanner'))
    from project_scanner import scan_project

    root      = sys.argv[1] if len(sys.argv) > 1 else "."
    scan      = scan_project(root)
    decisions = architect_dataset(scan.result_files, root)
    print_decisions(decisions)
    save_decisions(decisions, root)