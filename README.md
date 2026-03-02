# MAIRA — ML Agentic Intelligence for Research Automation

> Point MAIRA at any ML project folder. No code instrumentation needed.

MAIRA reads your folder structure, result files, and model checkpoints — then identifies experiment gaps, architects datasets, generates experiment scripts, and tracks what worked across runs.

---

## What MAIRA Does

| Step | What happens |
|------|-------------|
| Scan | Reads folder structure, result files, model checkpoints |
| Detect | Identifies project type — RL, CV, NLP, tabular |
| Gap analysis | Finds experiment combinations never tested |
| Dataset architect | Splits data with written justification |
| Research advisor | LLM-powered next-step recommendations |
| Code writer | Generates experiment scripts — human approves first |
| Memory | Tracks outcomes across runs, re-ranks suggestions |

**Original files are never modified.** All generated scripts go to `your_project/maira/generated/`

---

## Install
```bash
git clone https://github.com/yourusername/maira-ml
cd maira-ml
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

## LLM Provider Setup

MAIRA supports multiple LLM providers. On first run a setup wizard appears.

| Provider | Cost | GPU | Requests/day | Get key |
|----------|------|-----|--------------|---------|
| Groq | free tier | none | 14,400 | console.groq.com |
| Ollama | free | see below | unlimited | ollama.ai |
| Gemini | free tier | none | ~50/day | aistudio.google.com |
| Anthropic | paid | none | unlimited | console.anthropic.com |

### Recommended by hardware

| Hardware | Use |
|----------|-----|
| No GPU / any machine | Groq free tier |
| 4GB VRAM | Ollama mistral:7b-q4 |
| 6GB VRAM | Ollama llama3.2:3b |
| 8GB+ VRAM | Ollama llama3:8b |
| Privacy required | Ollama — fully local, no data leaves machine |
| Production / paid | Anthropic |

### Ollama setup (local, no internet)
```bash
# Install Ollama from ollama.ai
# Pull a model that fits your VRAM
ollama pull llama3.2:3b    # 2GB VRAM
ollama pull mistral:7b-q4  # 4GB VRAM

# Windows users — start with network access for WSL
$env:OLLAMA_HOST="0.0.0.0:11434"
ollama serve
```

---

## How MAIRA Finds Gaps

MAIRA detects gaps by analyzing folder names and experiment variants:
```
archived_models/latency/latency_10ms   ✓ tested
archived_models/latency/latency_50ms   ✓ tested
archived_models/latency/latency_100ms  ✓ tested
→ GAP: latency_25ms never tested
→ GAP: latency_200ms upper bound unknown

archived_models/baseline/ppo_full_obs  ✓ tested
archived_models/baseline/sac_full_obs  ✓ tested
→ GAP: TD3 never benchmarked against PPO baseline
```

---

## Feedback Memory

MAIRA remembers what it suggested across runs:
```bash
python maira/cli.py --status

  Total MAIRA runs:     3
  Total suggestions:    5
  Improved:             2
  No change:            1
  Degraded:             0
  Pending (not run):    2
```

After you run a generated experiment script, the next MAIRA scan automatically detects the new results, measures improvement against baseline, and re-ranks future suggestions.

---

## Project Structure
```
maira/
├── cli.py               # CLI entry point
├── __init__.py          # Python API
├── setup_wizard.py      # First-run provider setup
├── memory.py            # Feedback loop across runs
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

## What MAIRA Does Not Do

- Does not read or execute your code
- Does not modify any original project files
- Does not store API keys beyond your local machine
- Does not run experiments automatically — human approves first

---

## License

MIT