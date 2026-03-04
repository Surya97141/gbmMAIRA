"""
MAIRA v0.3 — Auto-detect Completed Runs
Watches the project for new result files that appeared
after a generated script was written.
Automatically closes the feedback loop without user intervention.
No other tool does this without instrumentation.
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────

@dataclass
class DetectedResult:
    gap:          str
    result_file:  str
    file_type:    str        # csv / npz / model
    detected_at:  str
    mean_reward:  Optional[float] = None
    delta:        Optional[float] = None
    outcome:      str        = "unknown"  # improved/degraded/no_change


# ─────────────────────────────────────────────────────────
# Match result files to gaps
# ─────────────────────────────────────────────────────────

RESULT_PATTERNS = {
    "latency_25ms":  ["latency_25ms", "latency25ms", "25ms"],
    "latency_200ms": ["latency_200ms", "latency200ms", "200ms"],
    "td3":           ["td3", "TD3"],
    "sac":           ["sac", "SAC"],
    "ppo":           ["ppo_new", "ppo_v2", "ppo_tuned"],
    "ensemble":      ["ensemble_new", "ensemble_v2"],
}


def _match_file_to_gap(filepath: str,
                        gaps: List[str]) -> Optional[str]:
    """Match a result file path to a gap by keyword matching."""
    fname = Path(filepath).stem.lower()

    for gap in gaps:
        gap_lower = gap.lower()

        # Direct keyword match
        keywords = gap_lower.replace(" ", "_").split("_")
        keywords = [k for k in keywords if len(k) > 2]

        matches = sum(1 for kw in keywords if kw in fname)
        if matches >= 2:
            return gap

        # Pattern matching
        for pattern_gap, patterns in RESULT_PATTERNS.items():
            if pattern_gap in gap_lower:
                if any(p.lower() in fname for p in patterns):
                    return gap

    return None


# ─────────────────────────────────────────────────────────
# Read reward from result file
# ─────────────────────────────────────────────────────────

def _read_reward_from_csv(filepath: str) -> Optional[float]:
    """Read mean reward from a CSV result file."""
    try:
        import pandas as pd
        df = pd.read_csv(filepath, header=None)
        if df.empty:
            return None
        # Generated scripts write: [timestep, mean_reward, std_reward]
        # Column 1 is mean_reward
        if df.shape[1] >= 2:
            return float(df.iloc[-1, 1])
        return float(df.iloc[-1, 0])
    except:
        pass

    # Fallback: try with headers
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
        for col in ["mean_reward", "reward", "eval_reward", "total_reward"]:
            if col in df.columns:
                return float(df[col].iloc[-1])
    except:
        pass

    return None


def _read_reward_from_npz(filepath: str) -> Optional[float]:
    """Read mean reward from an NPZ result file."""
    try:
        data = np.load(filepath)
        if "results" in data:
            return float(data["results"][-1].mean())
        if "rewards" in data:
            return float(data["rewards"][-1])
    except:
        pass
    return None


def _get_baseline_reward(project_root: str) -> Optional[float]:
    """Get baseline reward from ppo_logs."""
    try:
        npz = Path(project_root) / "ppo_logs" / "evaluations.npz"
        if npz.exists():
            data = np.load(npz)
            return float(data["results"][-1].mean())
    except:
        pass
    return None


# ─────────────────────────────────────────────────────────
# Load generated script registry
# ─────────────────────────────────────────────────────────

def _load_generated_registry(project_root: str) -> Dict:
    """
    Build a registry of generated scripts and when they were written.
    Returns: {gap_keyword: {"path": ..., "written_at": timestamp}}
    """
    registry = {}
    generated_dir = Path(project_root) / "maira" / "generated"

    if not generated_dir.exists():
        return registry

    for script in generated_dir.glob("exp_*.py"):
        stat        = script.stat()
        written_at  = stat.st_mtime
        name        = script.stem.lower()
        registry[name] = {
            "path":       str(script),
            "written_at": written_at,
        }

    return registry


# ─────────────────────────────────────────────────────────
# Core — scan for new result files
# ─────────────────────────────────────────────────────────

def auto_detect_completed_runs(project_root: str,
                                 gaps: List[str]) -> List[DetectedResult]:
    """
    Scan project for result files that appeared after
    generated scripts were written.
    Match them to gaps and measure reward delta.
    """
    root      = Path(project_root)
    registry  = _load_generated_registry(project_root)
    baseline  = _get_baseline_reward(project_root)
    detected  = []

    # Find the earliest script write time
    earliest_script_time = min(
        (v["written_at"] for v in registry.values()),
        default=0
    )

    # Scan for new CSV and NPZ files
    scan_patterns = [
        ("logs/metrics", "*.csv", "csv"),
        ("logs",         "*.csv", "csv"),
        ("results",      "*.csv", "csv"),
        ("ppo_logs",     "*.npz", "npz"),
        ("logs",         "*.npz", "npz"),
    ]

    found_files = set()

    for subdir, pattern, ftype in scan_patterns:
        scan_path = root / subdir
        if not scan_path.exists():
            continue

        for result_file in scan_path.glob(pattern):
            # Skip existing baseline files
            if result_file.name in ["evaluations.npz",
                                     "latency_sweep_v1.csv"]:
                continue

            # Skip empty files
            if result_file.stat().st_size == 0:
                continue

            # Check if file appeared after any script was written
            file_mtime = result_file.stat().st_mtime
            if file_mtime <= earliest_script_time:
                continue

            rel_path = str(result_file.relative_to(root))
            if rel_path in found_files:
                continue
            found_files.add(rel_path)

            # Match to a gap
            matched_gap = _match_file_to_gap(str(result_file), gaps)
            if not matched_gap:
                # Try matching against generated script names
                for script_name in registry:
                    if any(kw in result_file.stem.lower()
                           for kw in script_name.split("_")
                           if len(kw) > 3):
                        # Find the gap this script belongs to
                        for gap in gaps:
                            if any(kw in gap.lower()
                                   for kw in result_file.stem.lower().split("_")
                                   if len(kw) > 3):
                                matched_gap = gap
                                break
                        break

            if not matched_gap:
                continue

            # Read reward
            mean_reward = None
            if ftype == "csv":
                mean_reward = _read_reward_from_csv(str(result_file))
            elif ftype == "npz":
                mean_reward = _read_reward_from_npz(str(result_file))

            # Calculate delta vs baseline
            delta   = None
            outcome = "unknown"
            if mean_reward is not None and baseline is not None:
                delta = mean_reward - baseline
                if delta > 0.05 * abs(baseline):
                    outcome = "improved"
                elif delta < -0.05 * abs(baseline):
                    outcome = "degraded"
                else:
                    outcome = "no_change"
            elif mean_reward is not None:
                outcome = "completed"

            detected.append(DetectedResult(
                gap         = matched_gap,
                result_file = rel_path,
                file_type   = ftype,
                detected_at = datetime.now().isoformat(),
                mean_reward = mean_reward,
                delta       = delta,
                outcome     = outcome,
            ))

    return detected


# ─────────────────────────────────────────────────────────
# Update memory with detected results
# ─────────────────────────────────────────────────────────

def update_memory_with_detections(detected: List[DetectedResult],
                                   project_root: str) -> int:
    """
    Write detected results into MAIRA memory.
    Returns number of memory entries updated.
    """
    if not detected:
        return 0

    memory_path = Path(project_root).parent.parent
    # Find memory.json — it's in the MAIRA install dir
    # Try common locations
    possible_memory = [
        Path("/mnt/d/gbmMAIRA/maira/.memory.json"),
        Path(__file__).parent / ".memory.json",
    ]

    memory_file = None
    for p in possible_memory:
        if p.exists():
            memory_file = p
            break

    if not memory_file:
        return 0

    try:
        with open(memory_file) as f:
            memory = json.load(f)
    except:
        return 0

    updated = 0
    suggestions = memory.get("suggestions", [])

    for detection in detected:
        for suggestion in suggestions:
            if suggestion.get("gap") == detection.gap and \
               suggestion.get("outcome") == "pending":

                suggestion["outcome"]     = detection.outcome
                suggestion["result_file"] = detection.result_file
                suggestion["mean_reward"] = detection.mean_reward
                suggestion["delta"]       = detection.delta
                suggestion["detected_at"] = detection.detected_at
                updated += 1
                break

    if updated:
        with open(memory_file, "w") as f:
            json.dump(memory, f, indent=2)

    return updated


# ─────────────────────────────────────────────────────────
# Print
# ─────────────────────────────────────────────────────────

def print_auto_detect(detected: List[DetectedResult],
                       updated: int) -> None:
    print("\n" + "="*60)
    print("  MAIRA — Auto-detect Completed Runs")
    print("="*60)

    if not detected:
        print("\n  No new result files detected since last run.")
        print("  Run a generated script then re-run MAIRA.")
        print(f"\n  Watching: logs/metrics/*.csv  |  logs/*.npz\n")
        return

    print(f"\n  Detected {len(detected)} completed run(s)!\n")

    for d in detected:
        outcome_icon = {
            "improved":  "↑ IMPROVED",
            "degraded":  "↓ DEGRADED",
            "no_change": "→ NO CHANGE",
            "completed": "✓ COMPLETED",
            "unknown":   "? UNKNOWN",
        }.get(d.outcome, "? UNKNOWN")

        print(f"  [{outcome_icon}] {d.gap}")
        print(f"       File:   {d.result_file}")
        if d.mean_reward is not None:
            print(f"       Reward: {d.mean_reward:.3f}", end="")
            if d.delta is not None:
                sign = "+" if d.delta >= 0 else ""
                print(f"  (delta: {sign}{d.delta:.3f} vs baseline)", end="")
            print()
        print()

    if updated > 0:
        print(f"  Memory updated: {updated} suggestion(s) marked as complete.")
    print()


# ─────────────────────────────────────────────────────────
# Save detection log
# ─────────────────────────────────────────────────────────

def save_detection_log(detected: List[DetectedResult],
                        project_root: str) -> None:
    if not detected:
        return
    output = [
        {
            "gap":         d.gap,
            "result_file": d.result_file,
            "file_type":   d.file_type,
            "detected_at": d.detected_at,
            "mean_reward": d.mean_reward,
            "delta":       d.delta,
            "outcome":     d.outcome,
        }
        for d in detected
    ]
    path = Path(project_root) / "maira" / "maira_detections.json"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)