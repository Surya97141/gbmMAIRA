"""
MAIRA — CLI Entry Point
Usage:
  python maira/cli.py --scan /path/to/project
  python maira/cli.py --about
  python maira/cli.py --status
  python maira/cli.py --scan /path/to/project --reset

  After pip install:
  maira --scan /path/to/project
"""

import sys
import os
import click
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scanner'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dataset'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'advisor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'runner'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'reporter'))
sys.path.insert(0, os.path.dirname(__file__))

from project_scanner   import scan_project,       print_scan
from result_parser     import parse_all,           print_parsed
from schema_detector   import detect_schema,       print_schema
from dataset_architect import architect_dataset,   print_decisions, save_decisions
from llm_advisor       import get_advice,          print_advice,    save_advice
from report_generator  import save_report
from code_writer       import run_hitl_pipeline
from setup_wizard      import get_or_setup_config, load_config
from memory            import (record_run, record_suggestions,
                                measure_outcomes, rank_gaps_by_history,
                                print_memory_report)


def print_about():
    print("""
  MAIRA — ML Agentic Intelligence for Research Automation
  ========================================================

  MAIRA analyzes any ML project folder without touching your code.
  It reads folder structure, result files, and model checkpoints —
  then identifies experiment gaps, architects datasets, and generates
  experiment scripts for human review before anything runs.

  WHAT MAIRA DOES:
    1. Scans your project — finds result files, models, experiment variants
    2. Detects project type — RL, CV, NLP, tabular (auto-detected)
    3. Finds experiment gaps — what combinations were never tested
    4. Architects datasets — train/val/test splits with justification
    5. Advises next steps — LLM-powered research recommendations
    6. Writes experiment scripts — human approves before code is saved
    7. Tracks outcomes — remembers what worked across runs

  PROVIDER OPTIONS:
  ┌─────────────┬──────────────┬───────────┬───────────────┬────────────────┐
  │ Provider    │ Cost         │ GPU       │ Requests/day  │ Setup          │
  ├─────────────┼──────────────┼───────────┼───────────────┼────────────────┤
  │ Groq        │ free tier    │ none      │ 14,400        │ console.groq.com│
  │ Ollama 3B   │ free         │ 2GB VRAM  │ unlimited     │ ollama pull ... │
  │ Ollama 7B   │ free         │ 4GB VRAM  │ unlimited     │ ollama pull ... │
  │ Gemini      │ free tier    │ none      │ ~50           │ aistudio.google │
  │ Anthropic   │ paid         │ none      │ unlimited     │ console.anthropic│
  └─────────────┴──────────────┴───────────┴───────────────┴────────────────┘

  RECOMMENDED BY HARDWARE:
    No GPU / any machine  →  Groq free tier (recommended for most users)
    4GB VRAM              →  Ollama mistral:7b-q4
    6GB VRAM              →  Ollama llama3.2:3b
    8GB+ VRAM             →  Ollama llama3:8b
    Privacy required      →  Ollama (fully local, no data leaves machine)
    Paid / production     →  Anthropic

  ORIGINAL FILES: MAIRA never modifies your project files.
  All generated scripts go to: your_project/maira/generated/
""")


def print_status():
    config = load_config()
    print("\n" + "="*60)
    print("  MAIRA — Current Status")
    print("="*60)

    if not config:
        print("\n  Provider:  not configured")
        print("  Run MAIRA with --scan to start the setup wizard.\n")
    else:
        print(f"\n  Provider:  {config.get('provider', 'unknown')}")
        print(f"  Model:     {config.get('model', 'unknown')}")
        if config.get('ollama_host'):
            print(f"  Host:      {config.get('ollama_host')}")
        print()

    print_memory_report()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option('--scan',   default=None,  help='Path to ML project to analyze')
@click.option('--about',  is_flag=True,  help='What MAIRA is and provider comparison table')
@click.option('--status', is_flag=True,  help='Show current config and memory summary')
@click.option('--reset',  is_flag=True,  help='Reset LLM provider config')
def main(scan, about, status, reset):
    """
    MAIRA — ML Agentic Intelligence for Research Automation

    Analyzes any ML project folder. No code instrumentation needed.

    \b
    Examples:
      python maira/main.py --scan /path/to/project
      python maira/main.py --about
      python maira/main.py --status
      python maira/main.py --scan . --reset
    """

    # ── --about ──────────────────────────────────────
    if about:
        print_about()
        return

    # ── --status ─────────────────────────────────────
    if status:
        print_status()
        return

    # ── --scan required from here ────────────────────
    if not scan:
        click.echo("\n  Error: --scan is required to analyze a project.")
        click.echo("  Usage: python maira/main.py --scan /path/to/project")
        click.echo("  Help:  python maira/main.py --help")
        click.echo("  Info:  python maira/main.py --about\n")
        return

    print("\n  MAIRA — ML Agentic Intelligence for Research Automation")
    print("  " + "="*56)

    root = os.path.abspath(scan)

    # Reset config if requested
    if reset:
        config_path = os.path.join(os.path.dirname(__file__), ".config.json")
        if os.path.exists(config_path):
            os.remove(config_path)
            print("  Config reset.\n")

    # Load or create config via wizard
    config      = get_or_setup_config()
    provider    = config["provider"]
    api_key     = config.get("api_key")
    model       = config.get("model", "llama-3.3-70b-versatile")
    ollama_host = config.get("ollama_host", "http://localhost:11434")

    print(f"  Provider: {provider} | Model: {model}\n")

    # ── D1 — Scan + Parse + Schema ───────────────────
    project_scan = scan_project(root)
    print_scan(project_scan)

    parsed = parse_all(project_scan.result_files, root)
    print_parsed(parsed)

    schema = detect_schema(project_scan, parsed)
    print_schema(schema)

    # ── v0.2 — Dependency Graph ───────────────────────
    from dependency_graph import (build_dependency_graph,
                                   print_dependency_graph,
                                   save_dependency_graph,
                                   get_ready_gaps)
    dep_nodes = build_dependency_graph(schema.experiment_gap, root)
    print_dependency_graph(dep_nodes)
    save_dependency_graph(dep_nodes, root)

    # ── v0.2 — Hyperparameter Sensitivity Map ─────────
    from hyperparam_map import (build_hyperparam_map,
                                 print_hyperparam_map,
                                 save_hyperparam_map)
    hparam_results = build_hyperparam_map(
        root, project_scan.experiment_dirs
    )
    print_hyperparam_map(hparam_results)
    save_hyperparam_map(hparam_results, root)

    # ── v0.2 — Reward Curve Diagnosis ─────────────────
    from curve_diagnosis import (diagnose_curve,
                                  print_curve_diagnosis,
                                  save_curve_diagnosis)
    curve = diagnose_curve(root)
    print_curve_diagnosis(curve)
    save_curve_diagnosis(curve, root)

    # If curve is STILL_RISING or COLLAPSED — block benchmarks
    if curve and curve.shape in ["STILL_RISING", "COLLAPSED"]:
        schema.experiment_gap = [
            g for g in schema.experiment_gap
            if "td3" not in g.lower() and "benchmark" not in g.lower()
        ]

    # ── Feedback Memory — measure past outcomes ───────

    # Only pass READY gaps to advisor and code writer
    ready_gaps = get_ready_gaps(dep_nodes)
    if ready_gaps:
        schema.experiment_gap = ready_gaps
    # If all blocked — keep all gaps so pipeline doesn't break
    # but warn the user
    if not ready_gaps:
        print("  Warning: all gaps blocked by preconditions.")
        print("  Showing all gaps — fix blockers before running.\n")

    # ── Feedback Memory — measure past outcomes ───────
    updates = measure_outcomes(root)
    if updates:
        print(f"  Feedback: {len(updates)} suggestion(s) now have results.")
        for u in updates:
            print(f"    [{u['outcome'].upper()}] {u['gap']} — delta={u['delta']}")
        print()

    # ── Rank gaps using memory ────────────────────────
    schema.experiment_gap = rank_gaps_by_history(schema.experiment_gap)

    # ── Print memory report ───────────────────────────
    print_memory_report()

    # ── D2 — Dataset Architect ────────────────────────
    decisions = architect_dataset(project_scan.result_files, root)
    print_decisions(decisions)
    save_decisions(decisions, root)

    # ── D4 — Research Advisor ─────────────────────────
    advice = get_advice(
        schema, decisions, parsed,
        api_key     = api_key,
        provider    = provider,
        model       = model,
        ollama_host = ollama_host
    )
    print_advice(advice)
    save_advice(advice, root)

    # ── D5 — Report ───────────────────────────────────
    save_report(root)

    # ── L1 — Human-in-the-Loop Code Writer ────────────
    written = run_hitl_pipeline(
        schema, project_scan, root,
        provider    = provider,
        api_key     = api_key,
        model       = model,
        ollama_host = ollama_host
    )

    # ── Record this run in memory ─────────────────────
    all_gaps      = schema.experiment_gap
    approved_gaps = all_gaps if written else []
    record_suggestions(all_gaps, approved_gaps, root)
    record_run(root, all_gaps, approved_gaps, provider)

    print("  MAIRA complete.\n")


if __name__ == "__main__":
    main()