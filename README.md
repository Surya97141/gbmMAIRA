<p align="center">
  <img src="assets/gbmlg.jpeg" width="720" alt="MAIRA — Point · Scan · Discover · Build"/>
</p>

# MAIRA — ML Agentic Intelligence for Research Automation

> Point. Scan. Discover. Build.
> Point MAIRA at any ML project folder. No instrumentation needed.

MAIRA reads your folder structure, result files, and model checkpoints — then identifies experiment gaps, architects datasets, diagnoses training curves, scores reproducibility, generates experiment scripts, and automatically closes the feedback loop when experiments complete.

**"Copilot completes your code. MAIRA completes your research."**

---

## What Makes MAIRA Different

**Zero instrumentation.** Point MAIRA at any folder — no logging code, no tracking setup, no W&B account.
```bash
python maira/cli.py --scan /path/to/any/ml/project
```

**Domain-agnostic from v0.4.** Works on RL, CV, NLP, tabular ML, disaster prediction, time series — no hardcoded assumptions. The LLM figures out what kind of project it is from scratch.
```
Type:       REINFORCEMENT_LEARNING   ← aerial combat DRL project
Type:       TABULAR_ML               ← flood & disaster prediction project
Framework:  stable_baselines3 / scikit_learn  ← detected automatically
```

**Blocks invalid experiments before you waste compute.**
```
[✗] TD3 never benchmarked against PPO baseline
      BLOCKED: Evaluation protocol must match across algorithms
      FIX:     Use same n_eval_episodes, deterministic=True for all algorithms
```

**Diagnoses your training curve and tells you what to do.**
```
Shape:  ✓  HEALTHY — good
Steps:  500,000
Reward: start=3925  peak=3936  final=3671
→ Curve has converged. Safe to begin benchmarking experiments.
```

**Scores your research against publication standards — before you submit.**
```
Score:  10/14  [██████████████░░░░░░]  Grade: C
✗ Multiple seed runs missing  — single-seed results are not publishable
✗ Only 5 eval episodes        — reviewers will ask about this
✗ No README found             — required for reproducibility
```

**Automatically closes the feedback loop when experiments complete.**
```
[↑ IMPROVED] latency_25ms never tested — gap between 10ms and 50ms
     File:   logs/metrics/latency_25ms.csv
     Reward: 3975.800  (delta: +304.517 vs baseline)
     Memory updated: 1 suggestion marked as complete.
```

**Remembers everything across runs — longitudinal research memory.**
```
Total MAIRA runs:    11
Improved:             1
Pending (not run):    2
```

**Works fully local with Ollama — no data leaves your machine.**
```bash
ollama pull llama3.2:3b
python maira/cli.py --scan /path/to/project  # zero internet required
```

---

## Full Pipeline — What MAIRA Does

| Step | What happens |
|------|-------------|
| **Scan** | Reads folder structure, result files, model checkpoints. Skips venvs automatically. |
| **Parse** | Reads top 8 result files (NPZ → CSV → JSON priority) — prevents token overflow on large projects |
| **Analyze** | LLM identifies project type, framework, baseline metric, gaps — no hardcoded rules **(v0.4)** |
| **Dependency graph** | Blocks experiments whose preconditions aren't met |
| **Hyperparam map** | Correlates hyperparameter variants with outcome deltas |
| **Curve diagnosis** | Diagnoses training curve — PLATEAUED / STILL_RISING / COLLAPSED / OSCILLATING |
| **Reproducibility score** | Scores project against publication standards — Grade A to F |
| **Dataset architect** | Splits data with written justification |
| **Research advisor** | LLM-powered next-step recommendations |
| **Code writer** | Generates experiment scripts — human approves first **(v0.4: no LLM previews, direct to gate)** |
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
scripts = maira.generate(result, gaps=[result.gaps[0]])

# Configure provider programmatically
maira.configure("groq", api_key="gsk_...")
```

---

## What the Output Looks Like

```
============================================================
  MAIRA — Project Analysis                          [v0.4]
============================================================

  Type:       REINFORCEMENT_LEARNING
  Task:       single_agent_rl
  Framework:  stable_baselines3
  Confidence: HIGH

  Baseline metric: results in ppo_logs/evaluations.npz

  Experiment gaps (3 found):
    [HIGH  ] multiple_seeds
    [MEDIUM] ppo_vs_maddpg
    [LOW   ] sweep_of_hyperparameters

============================================================
  MAIRA — Training Curve Diagnosis
============================================================

  Shape:     ✓  HEALTHY — good
  Steps:     500,000
  Reward:    start=3925.962  peak=3936.495  final=3671.283
  Ready for evaluation: YES

============================================================
  MAIRA — Reproducibility Score
============================================================

  Score:   10/14  [██████████████░░░░░░]
  Grade:   C

  [✓] +2/2  Random seed fixed
  [✗] +0/2  Multiple seed runs (≥2)  ← not publishable without this
  [✓] +2/2  Baseline comparison exists

============================================================
  MAIRA — Experiment Possibilities
============================================================

  Found 3 experiment gap(s):

  [1] multiple_seeds
  [2] ppo_vs_maddpg
  [3] sweep_of_hyperparameters

  Your choice: 1

  Writing: multiple_seeds
  Saved:   maira/generated/exp_multiple_seeds_20260306_095228.py
```

---

## Token Limits and Large Projects

MAIRA sends a summary of your result files to the LLM. On projects with many large CSVs (20+ files with wide column schemas), this can exceed free-tier limits.

MAIRA automatically caps files sent to the LLM at 8, prioritising `.npz` → `.csv` → `.json`.

| Provider | Token limit | Suitable for |
|----------|------------|--------------|
| Groq free | 12,000 TPM | Most ML projects — RL, CV, NLP, small tabular |
| Groq paid | 100,000+ TPM | Any project, no file capping needed |
| Anthropic | 200,000 context | Any project, highest quality analysis |
| Ollama (local) | RAM-dependent | Unlimited, fully private |
| Gemini Pro | 1M context | Any project size |

---

## Reward / Training Curve Diagnosis

| Shape | Meaning | Action |
|-------|---------|--------|
| PLATEAUED | Converged — ready for benchmarking | Run algorithm comparisons now |
| STILL_RISING | Undertrained — don't benchmark yet | Train more steps first |
| COLLAPSED | Policy diverged | Reduce LR, use checkpoint before collapse |
| OSCILLATING | Unstable training | Increase batch size, reduce LR |
| HEALTHY | Improving, no issues | Continue or begin experiments |

---

## Reproducibility Score

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

## LLM Provider Setup

| Provider | Cost | GPU | Requests/day | Get key |
|----------|------|-----|--------------|---------|
| Groq | free tier | none | 14,400 | console.groq.com |
| Ollama | free | local | unlimited | ollama.ai |
| Gemini | free tier | none | ~50/day | aistudio.google.com |
| Anthropic | paid | none | unlimited | console.anthropic.com |

```bash
# Ollama local setup (Windows + WSL)
$env:OLLAMA_HOST="0.0.0.0:11434"
ollama serve
ollama pull llama3.2:3b
```

---

## Project Structure

```
maira/
├── cli.py                   # CLI entry point
├── __init__.py              # Python API
├── setup_wizard.py          # First-run provider setup
├── memory.py                # Feedback loop across runs
├── analyzer/
│   └── project_analyzer.py  # v0.4 — LLM project analysis
├── scanner/
│   ├── project_scanner.py   # Folder + file discovery (venv-aware)
│   ├── result_parser.py     # CSV / JSON / NPZ parsing
│   └── schema_detector.py   # Schema compatibility wrapper
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
v0.4  LLM-powered project analysis — no hardcoded domain rules
      Works on any ML domain: RL, CV, NLP, tabular, disaster prediction
      Virtual environment exclusion — skips venv folders automatically
      Token overflow protection — smart file cap before LLM call
      Approval gate fix — flush=True for WSL terminal compatibility
      Direct approval gate — no LLM previews, no hanging before gate appears

v0.3  Auto-detect completed runs
      Feedback loop closes without manual intervention
      Memory updates automatically when results appear

v0.2  Experiment dependency graph — blocks invalid experiments
      Hyperparameter sensitivity map
      Reward curve shape diagnosis
      Reproducibility score — Grade A to F

v0.1  Gap detection, dataset architect, research advisor,
      code writer + approval gate, memory, multi-provider LLM
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
