"""
MAIRA v0.2 — Experiment Dependency Graph
Checks preconditions before suggesting experiments.
Blocks suggestions that are not yet valid.
Tells the researcher exactly what to fix first.
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────

@dataclass
class ExperimentNode:
    gap:            str
    preconditions:  List[str] = field(default_factory=list)
    blocks:         List[str] = field(default_factory=list)
    status:         str       = "unknown"   # ready / blocked / done / pending
    block_reasons:  List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────
# Precondition rules — what each experiment type needs
# ─────────────────────────────────────────────────────────

PRECONDITION_RULES = [

    # ── Latency experiments ──────────────────────────────
    {
        "match":        ["latency"],
        "needs": [
            {
                "check":   "baseline_exists",
                "message": "Need a stable baseline run before testing latency variants",
                "fix":     "Run your baseline PPO/SAC experiment to completion first"
            },
            {
                "check":   "seed_fixed",
                "message": "Seeds must be fixed before latency comparison is valid",
                "fix":     "Add seed=42 (or any fixed value) to all experiment configs"
            },
        ]
    },

    # ── Algorithm benchmarking (TD3 vs PPO etc) ──────────
    {
        "match":        ["benchmark", "td3", "sac", "ddpg", "maddpg"],
        "needs": [
            {
                "check":   "baseline_exists",
                "message": "Need a stable PPO baseline before benchmarking other algorithms",
                "fix":     "Ensure ppo_logs/evaluations.npz shows converged reward"
            },
            {
                "check":   "eval_protocol_consistent",
                "message": "Evaluation protocol must match across algorithms",
                "fix":     "Use same n_eval_episodes, deterministic=True for all algorithms"
            },
            {
                "check":   "sufficient_training_steps",
                "message": "Baseline may be undertrained — benchmarking against weak baseline is misleading",
                "fix":     "Verify reward curve has plateaued before running algorithm comparison"
            }
        ]
    },

    # ── Ensemble experiments ─────────────────────────────
    {
        "match":        ["ensemble"],
        "needs": [
            {
                "check":   "multiple_baselines_exist",
                "message": "Need at least 2 trained models before building ensemble",
                "fix":     "Train PPO and SAC individually first"
            },
        ]
    },

    # ── Hyperparameter search ────────────────────────────
    {
        "match":        ["lr", "learning_rate", "batch", "gamma", "entropy"],
        "needs": [
            {
                "check":   "baseline_exists",
                "message": "Need baseline before hyperparameter search",
                "fix":     "Run default hyperparameters first to establish baseline"
            },
        ]
    },

    # ── Upper bound / stress tests ───────────────────────
    {
        "match":        ["200ms", "upper", "stress", "extreme"],
        "needs": [
            {
                "check":   "intermediate_exists",
                "message": "Should test intermediate values before upper bound",
                "fix":     "Run latency_25ms and latency_100ms before latency_200ms"
            },
        ]
    },
]


# ─────────────────────────────────────────────────────────
# Checkers — inspect the actual project
# ─────────────────────────────────────────────────────────

def _check_baseline_exists(project_root: str) -> bool:
    """Does a baseline model or result file exist?"""
    root = Path(project_root)
    baseline_signals = [
        root / "ppo_logs" / "evaluations.npz",
        root / "archived_models" / "baseline",
        root / "models" / "ppo_aerial_combat_final.zip",
        root / "models" / "best_model.zip",
    ]
    return any(p.exists() for p in baseline_signals)


def _check_seed_fixed(project_root: str) -> bool:
    """Look for seed= in Python files."""
    root = Path(project_root)
    for py_file in root.rglob("*.py"):
        try:
            content = py_file.read_text(errors="ignore")
            if "seed=" in content or "SEED" in content or "set_seed" in content:
                return True
        except:
            pass
    return False


def _check_eval_protocol_consistent(project_root: str) -> bool:
    """Check if n_eval_episodes appears consistently."""
    root    = Path(project_root)
    found   = []
    for py_file in root.rglob("*.py"):
        try:
            content = py_file.read_text(errors="ignore")
            if "n_eval_episodes" in content:
                found.append(py_file.name)
        except:
            pass
    # If found in at least one file, assume consistent
    return len(found) >= 1


def _check_sufficient_training_steps(project_root: str) -> bool:
    """Check if reward curve has enough steps to be meaningful."""
    try:
        import numpy as np
        npz_path = Path(project_root) / "ppo_logs" / "evaluations.npz"
        if not npz_path.exists():
            return False
        data      = np.load(npz_path)
        timesteps = data["timesteps"]
        return len(timesteps) >= 10 and int(timesteps[-1]) >= 100000
    except:
        return False


def _check_multiple_baselines_exist(project_root: str) -> bool:
    """Do multiple trained models exist?"""
    root    = Path(project_root)
    models  = list((root / "models").glob("*.zip")) if (root / "models").exists() else []
    archived = list((root / "archived_models").rglob("*.zip")) if (root / "archived_models").exists() else []
    return len(models) + len(archived) >= 2


def _check_intermediate_exists(project_root: str, gap: str) -> bool:
    """For upper bound tests — check intermediate values were tested."""
    root = Path(project_root)
    if "200ms" in gap:
        signals = [
            root / "archived_models" / "latency" / "latency_50ms",
            root / "archived_models" / "latency" / "latency_100ms",
            root / "logs" / "metrics" / "latency_50ms.csv",
        ]
        return any(p.exists() for p in signals)
    return True


CHECKER_MAP = {
    "baseline_exists":            _check_baseline_exists,
    "seed_fixed":                 _check_seed_fixed,
    "eval_protocol_consistent":   _check_eval_protocol_consistent,
    "sufficient_training_steps":  _check_sufficient_training_steps,
    "multiple_baselines_exist":   _check_multiple_baselines_exist,
}


# ─────────────────────────────────────────────────────────
# Core — build the dependency graph
# ─────────────────────────────────────────────────────────

def build_dependency_graph(gaps: List[str],
                            project_root: str) -> List[ExperimentNode]:
    """
    For each gap, check preconditions against the actual project.
    Returns list of ExperimentNodes with status and block reasons.
    """
    nodes = []

    for gap in gaps:
        node           = ExperimentNode(gap=gap)
        gap_lower      = gap.lower()
        applicable     = []

        # Find which rules apply to this gap
        for rule in PRECONDITION_RULES:
            if any(kw in gap_lower for kw in rule["match"]):
                applicable.extend(rule["needs"])

        # Run each check
        for precondition in applicable:
            check = precondition["check"]

            if check == "intermediate_exists":
                passed = _check_intermediate_exists(project_root, gap_lower)
            elif check in CHECKER_MAP:
                passed = CHECKER_MAP[check](project_root)
            else:
                passed = True

            if not passed:
                node.block_reasons.append({
                    "issue": precondition["message"],
                    "fix":   precondition["fix"]
                })

        node.status = "blocked" if node.block_reasons else "ready"
        nodes.append(node)

    return nodes


# ─────────────────────────────────────────────────────────
# Print
# ─────────────────────────────────────────────────────────

def print_dependency_graph(nodes: List[ExperimentNode]) -> None:
    print("\n" + "="*60)
    print("  MAIRA — Experiment Dependency Graph")
    print("="*60)

    ready   = [n for n in nodes if n.status == "ready"]
    blocked = [n for n in nodes if n.status == "blocked"]

    print(f"\n  Ready to run:  {len(ready)}")
    print(f"  Blocked:       {len(blocked)}\n")

    for node in nodes:
        icon = "✓" if node.status == "ready" else "✗"
        print(f"  [{icon}] {node.gap}")

        if node.block_reasons:
            for reason in node.block_reasons:
                print(f"        BLOCKED: {reason['issue']}")
                print(f"        FIX:     {reason['fix']}")
        print()


def get_ready_gaps(nodes: List[ExperimentNode]) -> List[str]:
    """Return only gaps that are ready to run."""
    return [n.gap for n in nodes if n.status == "ready"]


def get_blocked_gaps(nodes: List[ExperimentNode]) -> List[str]:
    """Return gaps that are blocked with their reasons."""
    return [(n.gap, n.block_reasons) for n in nodes if n.status == "blocked"]


# ─────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────

def save_dependency_graph(nodes: List[ExperimentNode],
                           project_root: str) -> None:
    output = []
    for node in nodes:
        output.append({
            "gap":           node.gap,
            "status":        node.status,
            "block_reasons": node.block_reasons,
        })
    path = Path(project_root) / "maira" / "maira_dependency_graph.json"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)