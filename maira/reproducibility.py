"""
MAIRA v0.2 — Reproducibility Score
Reads the project and scores it against publication standards.
Tells the researcher exactly what is missing and how to fix it.
No other tool does this automatically without instrumentation.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple
import re


# ─────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────

@dataclass
class ReproCheck:
    name:        str
    passed:      bool
    points:      int        # points awarded if passed
    max_points:  int        # max possible
    detail:      str        # what was found
    fix:         str        # what to do if failed


@dataclass
class ReproScore:
    total:       int
    max_total:   int
    grade:       str        # A / B / C / D / F
    verdict:     str        # publishable / needs work / not publishable
    checks:      List[ReproCheck] = field(default_factory=list)
    top_fixes:   List[str]        = field(default_factory=list)


# ─────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────

def _check_seed_fixed(root: Path) -> ReproCheck:
    """Are random seeds fixed in the code?"""
    found_seed = False
    seed_files = []
    for py in root.rglob("*.py"):
        try:
            content = py.read_text(errors="ignore")
            if re.search(r'seed\s*=\s*\d+|set_seed|np\.random\.seed|torch\.manual_seed|random\.seed', content):
                found_seed = True
                seed_files.append(py.name)
        except:
            pass

    return ReproCheck(
        name       = "Random seed fixed",
        passed     = found_seed,
        points     = 2 if found_seed else 0,
        max_points = 2,
        detail     = f"Seed found in: {', '.join(seed_files[:3])}" if found_seed
                     else "No seed= found in any Python file",
        fix        = "Add seed=42 to your training script and all eval scripts."
    )


def _check_eval_episodes(root: Path) -> ReproCheck:
    """Are enough eval episodes used?"""
    found_n    = None
    sufficient = False
    for py in root.rglob("*.py"):
        try:
            content = py.read_text(errors="ignore")
            match   = re.search(r'n_eval_episodes\s*=\s*(\d+)', content)
            if match:
                found_n    = int(match.group(1))
                sufficient = found_n >= 10
                break
        except:
            pass

    # Also check NPZ shape — results shape [n, episodes]
    try:
        import numpy as np
        npz = root / "ppo_logs" / "evaluations.npz"
        if npz.exists():
            data     = np.load(npz)
            n_eps    = data["results"].shape[1]
            found_n  = n_eps
            sufficient = n_eps >= 10
    except:
        pass

    return ReproCheck(
        name       = "Sufficient eval episodes (≥10)",
        passed     = sufficient,
        points     = 2 if sufficient else (1 if found_n else 0),
        max_points = 2,
        detail     = f"Found n_eval_episodes={found_n}" if found_n
                     else "n_eval_episodes not found",
        fix        = "Use at least 10 eval episodes for statistically meaningful results. "
                     "Reviewers will ask about this."
    )


def _check_multiple_seeds(root: Path) -> ReproCheck:
    """Evidence of multiple seed runs?"""
    seed_dirs  = []
    patterns   = ['seed_', 'seed-', 'run_', 'trial_']
    for folder in root.rglob("*"):
        if folder.is_dir():
            for pat in patterns:
                if pat in folder.name.lower():
                    seed_dirs.append(folder.name)
                    break

    found = len(seed_dirs) >= 2
    return ReproCheck(
        name       = "Multiple seed runs (≥2)",
        passed     = found,
        points     = 2 if found else 0,
        max_points = 2,
        detail     = f"Found {len(seed_dirs)} seed run directories: "
                     f"{', '.join(seed_dirs[:3])}" if found
                     else "No multiple-seed run directories found",
        fix        = "Run each experiment with at least 3 seeds and report mean ± std. "
                     "Single-seed results are not publishable."
    )


def _check_baseline_comparison(root: Path) -> ReproCheck:
    """Is there a baseline to compare against?"""
    has_baseline = (root / "archived_models" / "baseline").exists() or \
                   any(root.rglob("*baseline*"))

    return ReproCheck(
        name       = "Baseline comparison exists",
        passed     = has_baseline,
        points     = 2 if has_baseline else 0,
        max_points = 2,
        detail     = "Baseline directory found in archived_models/" if has_baseline
                     else "No baseline comparison found",
        fix        = "Train and save a random policy or simple baseline to compare against. "
                     "Without a baseline, improvement claims cannot be verified."
    )


def _check_eval_deterministic(root: Path) -> ReproCheck:
    """Is deterministic=True used in evaluation?"""
    found = False
    for py in root.rglob("*.py"):
        try:
            content = py.read_text(errors="ignore")
            if "deterministic=True" in content or "deterministic = True" in content:
                found = True
                break
        except:
            pass

    return ReproCheck(
        name       = "Deterministic evaluation",
        passed     = found,
        points     = 1 if found else 0,
        max_points = 1,
        detail     = "deterministic=True found in eval code" if found
                     else "deterministic=True not found",
        fix        = "Add deterministic=True to EvalCallback or evaluate() calls. "
                     "Stochastic evaluation makes results non-reproducible."
    )


def _check_requirements_file(root: Path) -> ReproCheck:
    """Is there a requirements.txt or environment file?"""
    has_req = (root / "requirements.txt").exists() or \
              (root / "environment.yml").exists() or \
              (root / "pyproject.toml").exists() or \
              (root / "setup.py").exists()

    return ReproCheck(
        name       = "Dependencies documented",
        passed     = has_req,
        points     = 1 if has_req else 0,
        max_points = 1,
        detail     = "requirements.txt or environment file found" if has_req
                     else "No requirements.txt or environment.yml found",
        fix        = "Add requirements.txt with pinned versions. "
                     "Run: pip freeze > requirements.txt"
    )


def _check_model_checkpoints(root: Path) -> ReproCheck:
    """Are model checkpoints saved?"""
    checkpoints = list(root.rglob("*checkpoint*")) + \
                  list(root.rglob("*best_model*"))
    found = len(checkpoints) >= 1

    return ReproCheck(
        name       = "Model checkpoints saved",
        passed     = found,
        points     = 1 if found else 0,
        max_points = 1,
        detail     = f"Found {len(checkpoints)} checkpoint file(s)" if found
                     else "No checkpoint files found",
        fix        = "Save model checkpoints during training so experiments can be reproduced. "
                     "Use CheckpointCallback with save_freq."
    )


def _check_config_documented(root: Path) -> ReproCheck:
    """Are hyperparameters documented somewhere?"""
    config_files = list(root.rglob("*.yaml")) + \
                   list(root.rglob("*.yml"))  + \
                   list(root.rglob("*config*"))
    found = len(config_files) >= 1

    return ReproCheck(
        name       = "Hyperparameters documented",
        passed     = found,
        points     = 1 if found else 0,
        max_points = 1,
        detail     = f"Found {len(config_files)} config file(s)" if found
                     else "No config or YAML files found",
        fix        = "Save hyperparameters to a config.yaml file before each run. "
                     "Reviewers need to know exact settings used."
    )


def _check_eval_log_exists(root: Path) -> ReproCheck:
    """Is there evaluation log data?"""
    has_eval = (root / "ppo_logs" / "evaluations.npz").exists() or \
               len(list(root.rglob("evaluations*"))) > 0 or \
               len(list(root.rglob("*eval_log*"))) > 0

    return ReproCheck(
        name       = "Evaluation logs saved",
        passed     = has_eval,
        points     = 1 if has_eval else 0,
        max_points = 1,
        detail     = "evaluations.npz found" if has_eval
                     else "No evaluation log files found",
        fix        = "Use EvalCallback with log_path set to save evaluation data."
    )


def _check_readme_exists(root: Path) -> ReproCheck:
    """Is there a README?"""
    has_readme = (root / "README.md").exists() or \
                 (root / "README.txt").exists() or \
                 (root / "README.rst").exists()

    return ReproCheck(
        name       = "README exists",
        passed     = has_readme,
        points     = 1 if has_readme else 0,
        max_points = 1,
        detail     = "README file found" if has_readme
                     else "No README found",
        fix        = "Add a README.md explaining how to run the experiments. "
                     "Required for anyone trying to reproduce your results."
    )


# ─────────────────────────────────────────────────────────
# Core — run all checks and compute score
# ─────────────────────────────────────────────────────────

def compute_reproducibility_score(project_root: str) -> ReproScore:
    root   = Path(project_root)
    checks = [
        _check_seed_fixed(root),
        _check_eval_episodes(root),
        _check_multiple_seeds(root),
        _check_baseline_comparison(root),
        _check_eval_deterministic(root),
        _check_requirements_file(root),
        _check_model_checkpoints(root),
        _check_config_documented(root),
        _check_eval_log_exists(root),
        _check_readme_exists(root),
    ]

    total     = sum(c.points     for c in checks)
    max_total = sum(c.max_points for c in checks)
    pct       = total / max_total if max_total > 0 else 0

    # Grade
    if pct >= 0.90:
        grade   = "A"
        verdict = "Publication ready"
    elif pct >= 0.75:
        grade   = "B"
        verdict = "Nearly publication ready — fix 1-2 items"
    elif pct >= 0.60:
        grade   = "C"
        verdict = "Needs work before submission"
    elif pct >= 0.40:
        grade   = "D"
        verdict = "Significant reproducibility gaps"
    else:
        grade   = "F"
        verdict = "Not reproducible — major issues"

    # Top fixes — failed checks sorted by max_points desc
    failed   = [c for c in checks if not c.passed]
    failed.sort(key=lambda c: c.max_points, reverse=True)
    top_fixes = [c.fix for c in failed[:4]]

    return ReproScore(
        total     = total,
        max_total = max_total,
        grade     = grade,
        verdict   = verdict,
        checks    = checks,
        top_fixes = top_fixes,
    )


# ─────────────────────────────────────────────────────────
# Print
# ─────────────────────────────────────────────────────────

def print_reproducibility_score(score: ReproScore) -> None:
    print("\n" + "="*60)
    print("  MAIRA — Reproducibility Score")
    print("="*60)

    bar_filled = int((score.total / score.max_total) * 20)
    bar        = "█" * bar_filled + "░" * (20 - bar_filled)

    print(f"\n  Score:   {score.total}/{score.max_total}  [{bar}]")
    print(f"  Grade:   {score.grade}")
    print(f"  Verdict: {score.verdict}\n")

    print("  Checks:")
    for c in score.checks:
        icon = "✓" if c.passed else "✗"
        pts  = f"+{c.points}/{c.max_points}"
        print(f"    [{icon}] {pts}  {c.name}")
        if not c.passed:
            print(f"           {c.detail}")

    if score.top_fixes:
        print("\n  Top fixes to improve score:")
        for i, fix in enumerate(score.top_fixes, 1):
            print(f"    {i}. {fix}")
    print()


# ─────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────

def save_reproducibility_score(score: ReproScore,
                                project_root: str) -> None:
    output = {
        "total":    score.total,
        "max":      score.max_total,
        "grade":    score.grade,
        "verdict":  score.verdict,
        "checks": [
            {
                "name":   c.name,
                "passed": c.passed,
                "points": c.points,
                "detail": c.detail,
                "fix":    c.fix,
            }
            for c in score.checks
        ],
        "top_fixes": score.top_fixes,
    }
    path = Path(project_root) / "maira" / "maira_reproducibility.json"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)