"""
MAIRA v0.2 — Reward Curve Shape Diagnosis
Reads evaluations.npz and diagnoses the training curve shape.
Tells you what the shape means and what to do next.
No other tool does this.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
import json


# ─────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────

@dataclass
class CurveDiagnosis:
    shape:           str            # PLATEAUED / STILL_RISING / COLLAPSED /
                                    # OSCILLATING / INSUFFICIENT_DATA / HEALTHY
    confidence:      str            # HIGH / MEDIUM / LOW
    final_reward:    float
    peak_reward:     float
    start_reward:    float
    total_steps:     int
    plateau_step:    Optional[int]  = None
    collapse_step:   Optional[int]  = None
    trend_slope:     float          = 0.0
    variance:        float          = 0.0
    insights:        List[str]      = field(default_factory=list)
    actions:         List[str]      = field(default_factory=list)
    ready_for_eval:  bool           = False


# ─────────────────────────────────────────────────────────
# Core diagnosis
# ─────────────────────────────────────────────────────────

def diagnose_curve(project_root: str) -> Optional[CurveDiagnosis]:
    """
    Load evaluations.npz and diagnose the reward curve shape.
    Returns None if no curve data found.
    """
    npz_path = Path(project_root) / "ppo_logs" / "evaluations.npz"
    if not npz_path.exists():
        return None

    try:
        data      = np.load(npz_path)
        timesteps = data["timesteps"]
        results   = data["results"]   # shape: [n_checkpoints, n_eval_episodes]
    except Exception as e:
        return None

    if len(timesteps) < 3:
        return CurveDiagnosis(
            shape        = "INSUFFICIENT_DATA",
            confidence   = "LOW",
            final_reward = float(results[-1].mean()) if len(results) else 0.0,
            peak_reward  = float(results.max()),
            start_reward = float(results[0].mean()),
            total_steps  = int(timesteps[-1]) if len(timesteps) else 0,
            insights     = ["Not enough checkpoints to diagnose curve shape.",
                            "Need at least 5 evaluation checkpoints."],
            actions      = ["Continue training and re-run MAIRA."],
            ready_for_eval = False
        )

    # Compute mean reward per checkpoint
    mean_rewards = results.mean(axis=1)
    n            = len(mean_rewards)

    final_reward = float(mean_rewards[-1])
    peak_reward  = float(mean_rewards.max())
    start_reward = float(mean_rewards[0])
    total_steps  = int(timesteps[-1])
    peak_idx     = int(mean_rewards.argmax())

    # Variance — measures oscillation
    variance = float(mean_rewards.std())

    # Trend slope — overall direction
    x           = np.arange(n)
    coeffs      = np.polyfit(x, mean_rewards, 1)
    trend_slope = float(coeffs[0])

    # Late slope — last 30% of training
    late_start  = int(n * 0.7)
    late_x      = np.arange(n - late_start)
    late_y      = mean_rewards[late_start:]
    late_coeffs = np.polyfit(late_x, late_y, 1) if len(late_y) > 1 else [0, 0]
    late_slope  = float(late_coeffs[0])

    # ── Detect COLLAPSE ──────────────────────────────────
    # Reward dropped > 20% from peak and never recovered
    drop_from_peak = (peak_reward - final_reward) / (abs(peak_reward) + 1e-8)
    collapse_step  = None

    if drop_from_peak > 0.20 and peak_idx < n - 2:
        # Find where collapse started
        for i in range(peak_idx, n):
            if mean_rewards[i] < peak_reward * 0.80:
                collapse_step = int(timesteps[i])
                break

        return CurveDiagnosis(
            shape        = "COLLAPSED",
            confidence   = "HIGH" if drop_from_peak > 0.40 else "MEDIUM",
            final_reward = final_reward,
            peak_reward  = peak_reward,
            start_reward = start_reward,
            total_steps  = total_steps,
            collapse_step = collapse_step,
            trend_slope  = trend_slope,
            variance     = variance,
            insights = [
                f"Reward peaked at {peak_reward:.3f} then collapsed "
                f"to {final_reward:.3f} — a {drop_from_peak*100:.1f}% drop.",
                f"Collapse began around step {collapse_step:,}.",
                "Likely causes: learning rate too high, "
                "entropy coefficient too low, or policy divergence.",
            ],
            actions = [
                "Reduce learning rate by 10x and retrain from checkpoint.",
                "Add entropy regularization to prevent premature convergence.",
                f"Use checkpoint saved before step {collapse_step:,}.",
                "Consider adding gradient clipping: max_grad_norm=0.5",
            ],
            ready_for_eval = False
        )

    # ── Detect OSCILLATING ───────────────────────────────
    # High variance relative to mean reward range
    reward_range   = peak_reward - start_reward
    norm_variance  = variance / (abs(reward_range) + 1e-8)

    if norm_variance > 0.3 and abs(late_slope) < abs(trend_slope) * 0.3:
        return CurveDiagnosis(
            shape        = "OSCILLATING",
            confidence   = "MEDIUM",
            final_reward = final_reward,
            peak_reward  = peak_reward,
            start_reward = start_reward,
            total_steps  = total_steps,
            trend_slope  = trend_slope,
            variance     = variance,
            insights = [
                f"Reward oscillating with std={variance:.3f} — unstable training.",
                f"Overall trend is positive (slope={trend_slope:.4f}) "
                "but not converging cleanly.",
                "High variance suggests environment stochasticity "
                "or unstable hyperparameters.",
            ],
            actions = [
                "Increase n_eval_episodes from 5 to 20 to smooth evaluation.",
                "Reduce learning rate by 2-5x.",
                "Increase batch size to stabilize gradient updates.",
                "Try adding a reward normalization wrapper.",
            ],
            ready_for_eval = False
        )

    # ── Detect PLATEAUED ────────────────────────────────
    # Late slope near zero — reward stopped improving
    plateau_threshold = abs(trend_slope) * 0.15
    plateau_step      = None

    if abs(late_slope) < plateau_threshold or abs(late_slope) < 0.001:
        # Find where plateau started
        for i in range(int(n * 0.5), n):
            window = mean_rewards[i:]
            if window.std() < variance * 0.3:
                plateau_step = int(timesteps[i])
                break

        return CurveDiagnosis(
            shape        = "PLATEAUED",
            confidence   = "HIGH" if abs(late_slope) < 0.0005 else "MEDIUM",
            final_reward = final_reward,
            peak_reward  = peak_reward,
            start_reward = start_reward,
            total_steps  = total_steps,
            plateau_step = plateau_step,
            trend_slope  = trend_slope,
            variance     = variance,
            insights = [
                f"Reward plateaued at {final_reward:.3f} — training converged.",
                f"Late slope: {late_slope:.6f} — essentially flat.",
                f"Plateau detected around step "
                f"{plateau_step:,}." if plateau_step else
                "Plateau covers final training phase.",
                "This is a good sign — model has converged.",
            ],
            actions = [
                "Model is ready for benchmarking experiments.",
                "This is the right time to run TD3 vs PPO comparison.",
                "Safe to run latency sensitivity experiments now.",
                "Consider running 3 seeds to confirm convergence is stable.",
            ],
            ready_for_eval = True
        )

    # ── Detect STILL RISING ──────────────────────────────
    if late_slope > 0.001:
        pct_of_peak = final_reward / (abs(peak_reward) + 1e-8)
        return CurveDiagnosis(
            shape        = "STILL_RISING",
            confidence   = "HIGH" if late_slope > 0.005 else "MEDIUM",
            final_reward = final_reward,
            peak_reward  = peak_reward,
            start_reward = start_reward,
            total_steps  = total_steps,
            trend_slope  = trend_slope,
            variance     = variance,
            insights = [
                f"Reward still improving — late slope: {late_slope:.6f}.",
                f"Current reward {final_reward:.3f} has not plateaued.",
                "Running experiments now would benchmark an undertrained model.",
                f"Estimated additional steps needed: "
                f"{int(total_steps * 0.5):,} more.",
            ],
            actions = [
                f"Train for at least {int(total_steps * 0.5):,} more steps.",
                "Re-run MAIRA after training to re-check curve.",
                "Do NOT run algorithm comparisons yet — baseline is undertrained.",
                "Latency experiments are OK — reward sensitivity is informative "
                "even before full convergence.",
            ],
            ready_for_eval = False
        )

    # ── Default: HEALTHY ────────────────────────────────
    return CurveDiagnosis(
        shape        = "HEALTHY",
        confidence   = "MEDIUM",
        final_reward = final_reward,
        peak_reward  = peak_reward,
        start_reward = start_reward,
        total_steps  = total_steps,
        trend_slope  = trend_slope,
        variance     = variance,
        insights     = [
            f"Reward improved from {start_reward:.3f} to {final_reward:.3f}.",
            "Curve shape looks healthy — no collapse or oscillation detected.",
        ],
        actions      = ["Continue training or begin benchmarking experiments."],
        ready_for_eval = True
    )


# ─────────────────────────────────────────────────────────
# ASCII curve visualisation
# ─────────────────────────────────────────────────────────

def _ascii_curve(mean_rewards: np.ndarray, width: int = 40,
                 height: int = 6) -> List[str]:
    """Draw a tiny ASCII reward curve."""
    mn  = mean_rewards.min()
    mx  = mean_rewards.max()
    rng = mx - mn if mx != mn else 1.0

    # Downsample to width
    indices = np.linspace(0, len(mean_rewards) - 1, width).astype(int)
    samples = mean_rewards[indices]

    # Normalise to height
    rows = []
    for row in range(height - 1, -1, -1):
        threshold = mn + (row / (height - 1)) * rng
        line = ""
        for val in samples:
            if val >= threshold:
                line += "█"
            else:
                line += " "
        rows.append(f"  │{line}│")

    rows.append(f"  └{'─' * width}┘")
    return rows


# ─────────────────────────────────────────────────────────
# Print
# ─────────────────────────────────────────────────────────

def print_curve_diagnosis(diagnosis: Optional[CurveDiagnosis]) -> None:
    print("\n" + "="*60)
    print("  MAIRA — Reward Curve Diagnosis")
    print("="*60)

    if diagnosis is None:
        print("\n  No evaluations.npz found — skipping curve diagnosis.\n")
        return

    shape_icons = {
        "PLATEAUED":         "◼  PLATEAUED      — converged",
        "STILL_RISING":      "↑  STILL RISING   — undertrained",
        "COLLAPSED":         "↓  COLLAPSED      — unstable",
        "OSCILLATING":       "~  OSCILLATING    — unstable",
        "HEALTHY":           "✓  HEALTHY        — good",
        "INSUFFICIENT_DATA": "?  INSUFFICIENT DATA",
    }

    print(f"\n  Shape:     {shape_icons.get(diagnosis.shape, diagnosis.shape)}")
    print(f"  Confidence:{diagnosis.confidence}")
    print(f"  Steps:     {diagnosis.total_steps:,}")
    print(f"  Reward:    start={diagnosis.start_reward:.3f}  "
          f"peak={diagnosis.peak_reward:.3f}  "
          f"final={diagnosis.final_reward:.3f}")

    if diagnosis.collapse_step:
        print(f"  Collapse at step: {diagnosis.collapse_step:,}")
    if diagnosis.plateau_step:
        print(f"  Plateau from step: {diagnosis.plateau_step:,}")

    # ASCII curve
    try:
        npz  = Path(diagnosis.__class__.__module__)  # placeholder
        data = np.load(
            str(Path(__file__).parent.parent /
                "ppo_logs" / "evaluations.npz")
        )
        mean_rewards = data["results"].mean(axis=1)
        print()
        for line in _ascii_curve(mean_rewards):
            print(line)
    except:
        pass

    print(f"\n  Ready for evaluation: "
          f"{'YES' if diagnosis.ready_for_eval else 'NO'}\n")

    if diagnosis.insights:
        print("  Insights:")
        for insight in diagnosis.insights:
            print(f"    → {insight}")

    print("\n  Recommended actions:")
    for action in diagnosis.actions:
        print(f"    • {action}")
    print()


# ─────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────

def save_curve_diagnosis(diagnosis: Optional[CurveDiagnosis],
                          project_root: str) -> None:
    if diagnosis is None:
        return
    output = {
        "shape":          diagnosis.shape,
        "confidence":     diagnosis.confidence,
        "final_reward":   diagnosis.final_reward,
        "peak_reward":    diagnosis.peak_reward,
        "start_reward":   diagnosis.start_reward,
        "total_steps":    diagnosis.total_steps,
        "plateau_step":   diagnosis.plateau_step,
        "collapse_step":  diagnosis.collapse_step,
        "ready_for_eval": diagnosis.ready_for_eval,
        "insights":       diagnosis.insights,
        "actions":        diagnosis.actions,
    }
    path = Path(project_root) / "maira" / "maira_curve_diagnosis.json"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)