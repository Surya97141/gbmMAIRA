"""
MAIRA v0.2 — Hyperparameter Sensitivity Map
Reads archived experiment variants and correlates
hyperparameter changes with outcome differences.
Answers: which hyperparameters actually matter in YOUR project?
"""

import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────

@dataclass
class HparamVariant:
    name:         str
    path:         str
    hparams:      Dict   = field(default_factory=dict)
    mean_reward:  Optional[float] = None
    final_reward: Optional[float] = None
    n_steps:      Optional[int]   = None


@dataclass
class SensitivityResult:
    param:        str
    values_seen:  List
    best_value:   object
    worst_value:  object
    delta:        float          # best - worst reward
    sensitivity:  str            # HIGH / MEDIUM / LOW
    insight:      str


# ─────────────────────────────────────────────────────────
# Extract hparams from folder names and config files
# ─────────────────────────────────────────────────────────

KNOWN_HPARAM_PATTERNS = {
    "latency":      r"latency[_-](\d+)ms",
    "lr":           r"lr[_-]([0-9e\-\.]+)",
    "batch":        r"batch[_-](\d+)",
    "gamma":        r"gamma[_-]([0-9\.]+)",
    "entropy":      r"entropy[_-]([0-9\.]+)",
    "n_steps":      r"nsteps[_-](\d+)",
    "layers":       r"layers[_-](\d+)",
    "algorithm":    r"(ppo|sac|td3|ddpg|maddpg|a2c)",
    "obs_type":     r"(full_obs|partial_obs|degraded)",
    "ensemble":     r"(ensemble[_\w]*)",
}


def _extract_hparams_from_name(folder_name: str) -> Dict:
    """Extract hyperparameter values from folder name patterns."""
    import re
    hparams = {}
    name_lower = folder_name.lower()

    for param, pattern in KNOWN_HPARAM_PATTERNS.items():
        match = re.search(pattern, name_lower)
        if match:
            hparams[param] = match.group(1)

    return hparams


def _read_reward_from_path(variant_path: str,
                            project_root: str) -> Optional[float]:
    """
    Try to find reward data for this variant.
    Checks NPZ, CSV, and JSON files in/near the variant folder.
    """
    path = Path(variant_path)

    # Check for evaluations.npz inside variant folder
    for npz in path.rglob("*.npz"):
        try:
            data = np.load(npz)
            if "results" in data:
                return float(data["results"][-1].mean())
        except:
            pass

    # Check for CSV with reward column
    for csv in path.rglob("*.csv"):
        try:
            import pandas as pd
            df = pd.read_csv(csv)
            for col in ["mean_reward", "reward", "eval_reward"]:
                if col in df.columns and not df[col].empty:
                    return float(df[col].iloc[-1])
        except:
            pass

    return None


def _get_baseline_reward(project_root: str) -> Optional[float]:
    """Get the baseline reward from ppo_logs."""
    try:
        npz = Path(project_root) / "ppo_logs" / "evaluations.npz"
        if npz.exists():
            data = np.load(npz)
            return float(data["results"][-1].mean())
    except:
        pass
    return None


# ─────────────────────────────────────────────────────────
# Core — scan experiment variants and build sensitivity map
# ─────────────────────────────────────────────────────────

def build_hyperparam_map(project_root: str,
                          experiment_dirs: List[str]) -> List[SensitivityResult]:
    """
    Scan all archived experiment variants.
    Extract hyperparameters from folder names.
    Correlate with reward outcomes.
    Return ranked sensitivity results.
    """
    root     = Path(project_root)
    variants = []

    # Scan archived_models and any experiment dirs
    scan_dirs = [root / "archived_models"]
    for d in experiment_dirs:
        p = root / d
        if p.exists():
            scan_dirs.append(p)

    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for folder in scan_dir.rglob("*"):
            if not folder.is_dir():
                continue
            hparams = _extract_hparams_from_name(folder.name)
            if not hparams:
                continue
            reward = _read_reward_from_path(str(folder), project_root)
            variants.append(HparamVariant(
                name        = folder.name,
                path        = str(folder),
                hparams     = hparams,
                mean_reward = reward,
            ))

    if not variants:
        return []

    # Add baseline as reference
    baseline_reward = _get_baseline_reward(project_root)

    # Group variants by hyperparameter
    param_groups: Dict[str, List[Tuple]] = {}
    for v in variants:
        for param, value in v.hparams.items():
            if param not in param_groups:
                param_groups[param] = []
            param_groups[param].append((value, v.mean_reward, v.name))

    # Calculate sensitivity per param
    results = []
    for param, entries in param_groups.items():
        # Filter entries that have reward data
        with_rewards = [(val, rew, name)
                        for val, rew, name in entries
                        if rew is not None]

        if len(with_rewards) < 2:
            # Not enough data — still report what we found
            values_seen = list(set(e[0] for e in entries))
            results.append(SensitivityResult(
                param       = param,
                values_seen = values_seen,
                best_value  = values_seen[0] if values_seen else "unknown",
                worst_value = "unknown",
                delta       = 0.0,
                sensitivity = "UNKNOWN",
                insight     = f"Found {len(values_seen)} variant(s) of '{param}' "
                              f"but no reward data to compare. "
                              f"Values seen: {', '.join(str(v) for v in values_seen)}"
            ))
            continue

        # Sort by reward
        sorted_entries = sorted(with_rewards, key=lambda x: x[1], reverse=True)
        best_val,  best_rew,  best_name  = sorted_entries[0]
        worst_val, worst_rew, worst_name = sorted_entries[-1]
        delta = best_rew - worst_rew

        # Classify sensitivity
        if delta > 1.0:
            sensitivity = "HIGH"
        elif delta > 0.3:
            sensitivity = "MEDIUM"
        else:
            sensitivity = "LOW"

        # Build insight
        values_seen = list(set(e[0] for e in entries))

        if param == "latency":
            insight = (f"Latency {best_val}ms gives best reward ({best_rew:.3f}). "
                      f"Latency {worst_val}ms gives worst ({worst_rew:.3f}). "
                      f"Delta: {delta:.3f}. "
                      f"Recommendation: keep latency ≤ {best_val}ms.")
        elif param == "algorithm":
            insight = (f"{best_val.upper()} outperforms {worst_val.upper()} "
                      f"by {delta:.3f} reward in this environment. "
                      f"Recommendation: use {best_val.upper()} as primary algorithm.")
        elif param == "obs_type":
            insight = (f"{best_val} observation gives {delta:.3f} more reward "
                      f"than {worst_val}. "
                      f"Recommendation: use {best_val} for all experiments.")
        else:
            insight = (f"'{param}={best_val}' outperforms '{param}={worst_val}' "
                      f"by {delta:.3f} reward. "
                      f"Recommendation: use {param}={best_val}.")

        results.append(SensitivityResult(
            param       = param,
            values_seen = values_seen,
            best_value  = best_val,
            worst_value = worst_val,
            delta       = delta,
            sensitivity = sensitivity,
            insight     = insight,
        ))

    # Sort by sensitivity — HIGH first, then delta
    priority = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNKNOWN": 3}
    results.sort(key=lambda r: (priority.get(r.sensitivity, 3), -r.delta))

    return results


# ─────────────────────────────────────────────────────────
# Print
# ─────────────────────────────────────────────────────────

def print_hyperparam_map(results: List[SensitivityResult]) -> None:
    print("\n" + "="*60)
    print("  MAIRA — Hyperparameter Sensitivity Map")
    print("="*60)

    if not results:
        print("\n  No hyperparameter variants found to compare.")
        print("  MAIRA needs at least 2 experiment variants with")
        print("  different hyperparameter values to build a map.\n")
        return

    high   = [r for r in results if r.sensitivity == "HIGH"]
    medium = [r for r in results if r.sensitivity == "MEDIUM"]
    low    = [r for r in results if r.sensitivity == "LOW"]
    unk    = [r for r in results if r.sensitivity == "UNKNOWN"]

    print(f"\n  Parameters analyzed:  {len(results)}")
    print(f"  High sensitivity:     {len(high)}")
    print(f"  Medium sensitivity:   {len(medium)}")
    print(f"  Low sensitivity:      {len(low)}")
    if unk:
        print(f"  No reward data:       {len(unk)}")
    print()

    for r in results:
        bar = _sensitivity_bar(r.sensitivity)
        print(f"  {bar} {r.param.upper()}")
        print(f"       Values tested: {', '.join(str(v) for v in r.values_seen)}")
        if r.sensitivity != "UNKNOWN":
            print(f"       Best:  {r.best_value}  |  Worst: {r.worst_value}"
                  f"  |  Delta: {r.delta:.3f}")
        print(f"       → {r.insight}")
        print()


def _sensitivity_bar(sensitivity: str) -> str:
    bars = {
        "HIGH":    "[███ HIGH   ]",
        "MEDIUM":  "[██  MEDIUM ]",
        "LOW":     "[█   LOW    ]",
        "UNKNOWN": "[?   UNKNOWN]",
    }
    return bars.get(sensitivity, "[?]")


# ─────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────

def save_hyperparam_map(results: List[SensitivityResult],
                         project_root: str) -> None:
    output = []
    for r in results:
        output.append({
            "param":       r.param,
            "sensitivity": r.sensitivity,
            "delta":       r.delta,
            "best_value":  str(r.best_value),
            "worst_value": str(r.worst_value),
            "values_seen": [str(v) for v in r.values_seen],
            "insight":     r.insight,
        })
    path = Path(project_root) / "maira" / "maira_hyperparam_map.json"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Hyperparam map saved to maira_hyperparam_map.json")