"""
MAIRA — Feedback Memory
Tracks what MAIRA suggested, what was run, and whether it worked.
This is the feedback loop — what makes MAIRA learn across runs.
Stores everything in maira/.memory.json
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional


MEMORY_PATH = Path(__file__).parent / ".memory.json"


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

def _empty_memory() -> dict:
    return {
        "project_root":   "",
        "created_at":     "",
        "last_updated":   "",
        "suggestions":    [],   # what MAIRA suggested
        "outcomes":       [],   # what actually happened
        "run_history":    [],   # each MAIRA scan run
    }


def load_memory() -> dict:
    if MEMORY_PATH.exists():
        with open(MEMORY_PATH) as f:
            return json.load(f)
    return _empty_memory()


def save_memory(memory: dict) -> None:
    memory["last_updated"] = datetime.now().isoformat()
    with open(MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=2)


# ─────────────────────────────────────────────
# Recording suggestions
# ─────────────────────────────────────────────

def record_suggestions(gaps: List[str], approved: List[str],
                        project_root: str) -> None:
    """Called after approval gate — record what was approved."""
    memory = load_memory()

    if not memory["project_root"]:
        memory["project_root"] = project_root
        memory["created_at"]   = datetime.now().isoformat()

    for gap in gaps:
        # Check if already recorded
        existing = [s for s in memory["suggestions"] if s["gap"] == gap]
        if existing:
            continue

        memory["suggestions"].append({
            "gap":        gap,
            "suggested_at": datetime.now().isoformat(),
            "approved":   gap in approved,
            "outcome":    "pending",   # pending / improved / no_change / degraded
            "result_file": None,
            "delta":      None,        # improvement over baseline
        })

    save_memory(memory)


# ─────────────────────────────────────────────
# Measuring outcomes
# ─────────────────────────────────────────────

def _find_result_file(gap: str, project_root: str) -> Optional[str]:
    """Look for a CSV result file that matches this gap."""
    import re
    metrics_dir = Path(project_root) / "logs" / "metrics"
    if not metrics_dir.exists():
        return None

    # Build keywords from gap name
    keywords = []
    for word in gap.lower().replace("—", " ").replace("-", " ").split():
        if len(word) > 3:
            keywords.append(word)

    for csv_file in metrics_dir.glob("*.csv"):
        fname = csv_file.stem.lower()
        matches = sum(1 for kw in keywords if kw in fname)
        if matches >= 2:
            return str(csv_file)

    return None


def _read_mean_reward(csv_path: str) -> Optional[float]:
    """Read final mean reward from a CSV result file."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        # Try common column names
        for col in ['mean_reward', 'reward', 'mean_ep_reward', 'eval_reward']:
            if col in df.columns:
                return float(df[col].iloc[-1])
        # Try last numeric column
        numeric = df.select_dtypes(include='number')
        if not numeric.empty:
            return float(numeric.iloc[-1, -1])
    except:
        pass
    return None


def _get_baseline_reward(project_root: str) -> Optional[float]:
    """Get baseline mean reward from ppo_logs/evaluations.npz"""
    try:
        import numpy as np
        npz_path = Path(project_root) / "ppo_logs" / "evaluations.npz"
        if not npz_path.exists():
            return None
        data    = np.load(npz_path)
        results = data["results"]
        return float(results[-1].mean())
    except:
        return None


def measure_outcomes(project_root: str) -> List[dict]:
    """
    Scan for new result files and measure if suggestions worked.
    Returns list of outcome updates.
    """
    memory   = load_memory()
    baseline = _get_baseline_reward(project_root)
    updates  = []

    for suggestion in memory["suggestions"]:
        if suggestion["outcome"] != "pending":
            continue
        if not suggestion["approved"]:
            continue

        result_file = _find_result_file(suggestion["gap"], project_root)
        if not result_file:
            continue

        mean_reward = _read_mean_reward(result_file)
        if mean_reward is None:
            continue

        # Measure delta against baseline
        delta = None
        if baseline is not None:
            delta = round(mean_reward - baseline, 4)

        if delta is None:
            outcome = "inconclusive"
        elif delta > 0.05:
            outcome = "improved"
        elif delta < -0.05:
            outcome = "degraded"
        else:
            outcome = "no_change"

        suggestion["outcome"]     = outcome
        suggestion["result_file"] = result_file
        suggestion["delta"]       = delta
        suggestion["measured_at"] = datetime.now().isoformat()
        suggestion["mean_reward"] = mean_reward
        suggestion["baseline"]    = baseline

        updates.append(suggestion)

    save_memory(memory)
    return updates


# ─────────────────────────────────────────────
# Run history
# ─────────────────────────────────────────────

def record_run(project_root: str, gaps_found: List[str],
               gaps_approved: List[str], provider: str) -> None:
    """Record each MAIRA scan run."""
    memory = load_memory()
    memory["run_history"].append({
        "timestamp":     datetime.now().isoformat(),
        "project_root":  project_root,
        "gaps_found":    gaps_found,
        "gaps_approved": gaps_approved,
        "provider":      provider,
    })
    save_memory(memory)


# ─────────────────────────────────────────────
# Memory-aware gap ranking
# ─────────────────────────────────────────────

def rank_gaps_by_history(gaps: List[str]) -> List[str]:
    """
    Re-rank gaps using memory:
    - Gaps that previously improved → rank higher
    - Gaps that previously degraded → rank lower
    - New gaps → keep original order
    """
    memory  = load_memory()
    outcomes = {s["gap"]: s["outcome"] for s in memory["suggestions"]}

    priority = {"improved": 0, "pending": 1, "inconclusive": 2,
                "no_change": 3, "degraded": 4}

    def rank(gap):
        outcome = outcomes.get(gap, "pending")
        return priority.get(outcome, 1)

    return sorted(gaps, key=rank)


# ─────────────────────────────────────────────
# Print memory report
# ─────────────────────────────────────────────

def print_memory_report() -> None:
    memory = load_memory()

    print("\n" + "="*60)
    print("  MAIRA — Feedback Memory")
    print("="*60)

    runs = memory.get("run_history", [])
    print(f"\n  Total MAIRA runs:     {len(runs)}")
    print(f"  Total suggestions:    {len(memory['suggestions'])}")

    if not memory["suggestions"]:
        print("  No suggestions recorded yet.")
        print()
        return

    pending   = [s for s in memory["suggestions"] if s["outcome"] == "pending"]
    improved  = [s for s in memory["suggestions"] if s["outcome"] == "improved"]
    degraded  = [s for s in memory["suggestions"] if s["outcome"] == "degraded"]
    no_change = [s for s in memory["suggestions"] if s["outcome"] == "no_change"]

    print(f"  Improved:             {len(improved)}")
    print(f"  No change:            {len(no_change)}")
    print(f"  Degraded:             {len(degraded)}")
    print(f"  Pending (not run):    {len(pending)}")

    if improved or degraded or no_change:
        print("\n  Suggestion outcomes:")
        for s in memory["suggestions"]:
            if s["outcome"] == "pending":
                continue
            delta_str = f"delta={s['delta']:+.4f}" if s["delta"] is not None else ""
            print(f"    [{s['outcome'].upper():12}] {s['gap'][:50]} {delta_str}")

    if pending:
        print("\n  Still pending (approved but not run yet):")
        for s in pending:
            if s["approved"]:
                print(f"    {s['gap']}")

    print()