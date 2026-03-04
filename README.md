# MAIRA — ML Agentic Intelligence for Research Automation

> Point. Scan. Discover. Build.
> Point MAIRA at any ML project folder. No instrumentation needed.

MAIRA reads your folder structure, result files, and model checkpoints — then identifies experiment gaps, architects datasets, diagnoses training curves, scores reproducibility, generates experiment scripts, and automatically closes the feedback loop when experiments complete.

**"Copilot completes your code. MAIRA completes your research."**

---

## What Makes MAIRA Different

| Capability | Copilot | W&B / MLflow | Devin | MAIRA |
|---|---|---|---|---|
| Zero instrumentation required | ✗ | ✗ | ✗ | ✅ |
| Detects experiment gaps | ✗ | ✗ | ✗ | ✅ |
| Blocks invalid experiments | ✗ | ✗ | ✗ | ✅ |
| Reward curve diagnosis | ✗ | partial | ✗ | ✅ |
| Reproducibility score | ✗ | ✗ | ✗ | ✅ |
| Auto-detects completed runs | ✗ | requires logging | ✗ | ✅ |
| Longitudinal research memory | ✗ | ✗ | ✗ | ✅ |
| Works fully local / offline | ✗ | ✗ | ✗ | ✅ |

---

## Full Pipeline — What MAIRA Does

| Step | What happens |
|------|-------------|
| **Scan** | Reads folder structure, result files, model checkpoints |
| **Detect** | Identifies project type — RL, CV, NLP, tabular |
| **Gap analysis** | Finds experiment combinations never tested |
| **Dependency graph** | Blocks experiments whose preconditions aren't met |
| **Hyperparam map** | Correlates hyperparameter variants with reward outcomes |
| **Curve diagnosis** | Diagnoses reward curve — PLATEAUED / STILL_RISING / COLLAPSED / OSCILLATING |
| **Reproducibility score** | Scores project against publication standards — Grade A to F |
| **Dataset architect** | Splits data with written justification |
| **Research advisor** | LLM-powered next-step recommendations |
| **Code writer** | Generates experiment scripts — human approves first |
| **Auto-detect** | Watches for new result files, measures delta vs baseline automatically |
| **Memory** | Tracks outcomes across runs, re-ranks suggestions |

**Original files are never modified.** All generated scripts go to `your_project/maira/generated/`

---

## Install

```bash
git clone https://github.com/Surya97141/gbmMAIRA
cd gbmMAIRA
pip install -r requirements.txt
```

---

## CLI Usage

```bash
# Analyze any ML project
python maira/cli.py --scan /path/to/your/ml/project

# What MAIRA is + provider table
python maira/cli.py --about

# Current config + memory summary
python maira/cli.py --status

# Switch LLM provider
python maira/cli.py --scan /path/to/project --reset
```

---

## Python API

```python
import sys
sys.path.insert(0, 'maira')
import maira

# One liner — full pipeline
maira.run("/path/to/my/ml/project")

# Step by step
result  = maira.scan("/path/to/project")
print(result.project_type)   # REINFORCEMENT_LEARNING
print(result.gaps)           # ['latency_25ms never tested', ...]

advice  = maira.advise(result)
print(advice.raw_response)

scripts = maira.generate(result, gaps=[result.gaps[0]])
print(scripts)   # ['path/to/generated/exp_....py']

# Configure provider programmatically
maira.configure("groq", api_key="gsk_...")

# Check status
maira.status()
```

---

## What the Output Looks Like

```
============================================================
  MAIRA — Experiment Dependency Graph
============================================================

  Ready to run:  2
  Blocked:       1

  [✓] latency_25ms never tested — gap between 10ms and 50ms
  [✓] latency_200ms never tested — upper bound unknown
  [✗] TD3 never benchmarked against PPO baseline
        BLOCKED: Evaluation protocol must match across algorithms
        FIX:     Use same n_eval_episodes, deterministic=True for all algorithms

============================================================
  MAIRA — Reward Curve Diagnosis
============================================================

  Shape:     ✓  HEALTHY — good
  Steps:     500,000
  Reward:    start=3925.962  peak=3936.495  final=3671.283
  Ready for evaluation: YES

  → Curve shape looks healthy — no collapse or oscillation detected.
  • Continue training or begin benchmarking experiments.

============================================================
  MAIRA — Reproducibility Score
============================================================

  Score:   10/14  [██████████████░░░░░░]
  Grade:   C
  Verdict: Needs work before submission

  [✓] +2/2  Random seed fixed
  [✗] +0/2  Multiple seed runs (≥2)  ← not publishable without this
  [✓] +2/2  Baseline comparison exists
  [✓] +1/1  Deterministic evaluation
  [✗] +0/1  README exists

  Top fixes:
    1. Run each experiment with at least 3 seeds — report mean ± std
    2. Add a README.md explaining how to reproduce results

============================================================
  MAIRA — Auto-detect Completed Runs
============================================================

  Detected 1 completed run(s)!

  [↑ IMPROVED] latency_25ms never tested — gap between 10ms and 50ms
       File:   logs/metrics/latency_25ms.csv
       Reward: 3975.800  (delta: +304.517 vs baseline)

  Memory updated: 1 suggestion(s) marked as complete.
```

---

## Experiment Dependency Graph

MAIRA blocks experiments whose preconditions aren't met — so you don't waste compute on invalid comparisons.

```
Precondition rules built in:

  Latency experiments      → need: baseline exists, seeds fixed
  Algorithm benchmarking   → need: baseline exists, eval protocol consistent
  Ensemble experiments     → need: multiple baselines exist
  Hyperparameter search    → need: baseline exists
  Upper bound tests        → need: intermediate values tested first
```

---

## Reward Curve Diagnosis

MAIRA reads your `evaluations.npz` and tells you what shape your training curve is — and what to do about it.

| Shape | Meaning | Action |
|-------|---------|--------|
| PLATEAUED | Converged — ready for benchmarking | Run algorithm comparisons now |
| STILL_RISING | Undertrained — don't benchmark yet | Train more steps first |
| COLLAPSED | Policy diverged | Reduce LR, use checkpoint before collapse |
| OSCILLATING | Unstable training | Increase batch size, reduce LR |
| HEALTHY | Improving, no issues | Continue or begin experiments |

---

## Reproducibility Score

MAIRA scores your project against 10 publication criteria — before you submit.

```
Checks performed:
  ✓ Random seed fixed          (2 pts)
  ✓ Sufficient eval episodes   (2 pts)  — need ≥ 10
  ✓ Multiple seed runs         (2 pts)  — need ≥ 2 seeds
  ✓ Baseline comparison        (2 pts)
  ✓ Deterministic evaluation   (1 pt)
  ✓ Dependencies documented    (1 pt)
  ✓ Model checkpoints saved    (1 pt)
  ✓ Hyperparameters documented (1 pt)
  ✓ Evaluation logs saved      (1 pt)
  ✓ README exists              (1 pt)

  Total: 14 points  |  Grades: A (≥90%) B (≥75%) C (≥60%) D (≥40%) F (<40%)
```

---

## Auto-detect Completed Runs

When you run a generated experiment script and results appear, MAIRA automatically detects the new files on the next scan, matches them to the correct gap, measures the reward delta vs baseline, and updates memory — no manual intervention needed.

```
Generated script writes → logs/metrics/latency_25ms.csv
Next MAIRA scan detects → new file appeared after script was written
MAIRA measures         → reward delta vs ppo_logs/evaluations.npz baseline
Memory updates         → Pending → Improved
```

---

## How MAIRA Finds Gaps

```
archived_models/latency/latency_10ms   ✓ tested
archived_models/latency/latency_50ms   ✓ tested
archived_models/latency/latency_100ms  ✓ tested
→ GAP: latency_25ms never tested — gap between 10ms and 50ms
→ GAP: latency_200ms never tested — upper bound unknown

archived_models/baseline/ppo_full_obs  ✓ tested
archived_models/baseline/sac_full_obs  ✓ tested
→ GAP: TD3 never benchmarked against PPO baseline
```

---

## LLM Provider Setup

MAIRA supports multiple LLM providers. On first run a setup wizard appears.

| Provider | Cost | GPU | Requests/day | Get key |
|----------|------|-----|--------------|---------|
| Groq | free tier | none | 14,400 | console.groq.com |
| Ollama | free | local | unlimited | ollama.ai |
| Gemini | free tier | none | ~50/day | aistudio.google.com |
| Anthropic | paid | none | unlimited | console.anthropic.com |

### Recommended by hardware

| Hardware | Use |
|----------|-----|
| No GPU / any machine | Groq free tier |
| 4GB VRAM | Ollama mistral:7b-q4 |
| 8GB+ VRAM | Ollama llama3:8b |
| Privacy required | Ollama — fully local, no data leaves machine |
| Production | Anthropic |

### Ollama (local, no internet)

```bash
ollama pull llama3.2:3b    # 2GB VRAM
ollama pull mistral:7b-q4  # 4GB VRAM

# Windows + WSL
$env:OLLAMA_HOST="0.0.0.0:11434"
ollama serve
```

---

## Project Structure

```
maira/
├── cli.py                   # CLI entry point
├── __init__.py              # Python API
├── setup_wizard.py          # First-run provider setup
├── memory.py                # Feedback loop across runs
├── dependency_graph.py      # v0.2 — precondition checking
├── hyperparam_map.py        # v0.2 — sensitivity analysis
├── curve_diagnosis.py       # v0.2 — reward curve shape
├── reproducibility.py       # v0.2 — publication readiness
├── auto_detect.py           # v0.3 — detect completed runs
├── scanner/
│   ├── project_scanner.py   # Folder + file discovery
│   ├── result_parser.py     # CSV / JSON / NPZ parsing
│   └── schema_detector.py   # Project type + gap detection
├── dataset/
│   └── dataset_architect.py # Train/val/test splits
├── advisor/
│   └── llm_advisor.py       # LLM research recommendations
├── runner/
│   └── code_writer.py       # Human-in-the-loop code generation
└── reporter/
    └── report_generator.py  # Markdown report output
```

---

## Version History

```
v0.1  Gap detection, dataset architect, research advisor,
      code writer + approval gate, memory, multi-provider LLM

v0.2  Experiment dependency graph — blocks invalid experiments
      Hyperparameter sensitivity map
      Reward curve shape diagnosis
      Reproducibility score — Grade A to F

v0.3  Auto-detect completed runs
      Feedback loop closes without manual intervention
      Memory updates automatically when results appear
```

---

## What MAIRA Does Not Do

- Does not read or execute your code
- Does not modify any original project files
- Does not store API keys beyond your local machine
- Does not run experiments automatically — human approves first

---

## License

MIT
