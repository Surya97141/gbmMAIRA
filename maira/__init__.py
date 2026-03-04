"""
MAIRA — ML Agentic Intelligence for Research Automation
pip install maira-ml

Usage:
    import maira
    maira.run("/path/to/project")
"""

import os
import sys

# Make all submodules importable
_here = os.path.dirname(__file__)
for _sub in ['scanner', 'dataset', 'advisor', 'runner', 'reporter']:
    _path = os.path.join(_here, _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)
if _here not in sys.path:
    sys.path.insert(0, _here)

from project_scanner   import scan_project
from result_parser     import parse_all
from schema_detector   import detect_schema
from dataset_architect import architect_dataset, save_decisions
from llm_advisor       import get_advice, save_advice
from report_generator  import save_report
from code_writer       import run_hitl_pipeline
from setup_wizard      import get_or_setup_config, load_config
from memory            import (record_run, record_suggestions,
                                measure_outcomes, rank_gaps_by_history,
                                print_memory_report)



# Public API


def scan(path: str):
    """
    Scan an ML project folder.
    Returns a ScanResult with gaps, project type, result files.

    Example:
        import maira
        result = maira.scan("/path/to/project")
        print(result.project_type)
        print(result.gaps)
    """
    root         = os.path.abspath(path)
    project_scan = scan_project(root)
    parsed       = parse_all(project_scan.result_files, root)
    schema       = detect_schema(project_scan, parsed)

    from dependency_graph import build_dependency_graph, print_dependency_graph, get_ready_gaps
    dep_nodes  = build_dependency_graph(schema.experiment_gap, root)
    print_dependency_graph(dep_nodes)
    ready_gaps = get_ready_gaps(dep_nodes)
    if ready_gaps:
        schema.experiment_gap = ready_gaps

    # Attach useful attributes for API users
    schema.gaps          = schema.experiment_gap
    schema.scan          = project_scan
    schema.parsed        = parsed
    schema.root          = root

    return schema


def advise(scan_result, provider: str = None, api_key: str = None,
           model: str = None, ollama_host: str = None):
    """
    Get AI-powered research recommendations for a scanned project.

    Example:
        result = maira.scan("/path/to/project")
        advice = maira.advise(result, provider="groq", api_key="gsk_...")
        print(advice.raw_response)
    """
    config = load_config()

    provider    = provider    or config.get("provider",    "groq")
    api_key     = api_key     or config.get("api_key",     None)
    model       = model       or config.get("model",       "llama-3.3-70b-versatile")
    ollama_host = ollama_host or config.get("ollama_host", "http://localhost:11434")

    root      = scan_result.root
    decisions = architect_dataset(scan_result.scan.result_files, root)
    save_decisions(decisions, root)

    advice = get_advice(
        scan_result, decisions, scan_result.parsed,
        api_key     = api_key,
        provider    = provider,
        model       = model,
        ollama_host = ollama_host
    )
    save_advice(advice, root)
    return advice


def generate(scan_result, gaps: list = None, approve: bool = True,
             provider: str = None, api_key: str = None,
             model: str = None, ollama_host: str = None) -> list:
    """
    Generate experiment scripts for detected gaps.
    Human approval gate shown by default (approve=True).
    Pass approve=False to auto-generate without gate.

    Example:
        result  = maira.scan("/path/to/project")
        scripts = maira.generate(result, gaps=[result.gaps[0]])
        print(scripts)   # list of generated file paths
    """
    config = load_config()

    provider    = provider    or config.get("provider",    "groq")
    api_key     = api_key     or config.get("api_key",     None)
    model       = model       or config.get("model",       "llama-3.3-70b-versatile")
    ollama_host = ollama_host or config.get("ollama_host", "http://localhost:11434")

    # Override gaps if specified
    if gaps:
        scan_result.experiment_gap = gaps

    if not approve:
        # Auto-approve all — no human gate
        from code_writer import write_approved
        output_dir = os.path.join(scan_result.root, "maira", "generated")
        return write_approved(
            scan_result.experiment_gap, scan_result,
            scan_result.root, provider, api_key, model, ollama_host
        )

    return run_hitl_pipeline(
        scan_result, scan_result.scan,
        scan_result.root, provider, api_key, model, ollama_host
    )


def configure(provider: str, api_key: str = None,
              model: str = None, ollama_host: str = None) -> None:
    """
    Set the LLM provider programmatically.
    Saves to maira/.config.json for future runs.

    Example:
        maira.configure("groq", api_key="gsk_...",
                         model="llama-3.3-70b-versatile")
    """
    from setup_wizard import save_config
    config = {
        "provider":    provider,
        "api_key":     api_key,
        "model":       model or "llama-3.3-70b-versatile",
        "ollama_host": ollama_host or "http://localhost:11434"
    }
    save_config(config)
    print(f"  MAIRA configured: {provider} / {config['model']}")


def status() -> None:
    """
    Print current provider config and memory summary.

    Example:
        maira.status()
    """
    config = load_config()
    print("\n  Provider:", config.get("provider", "not configured"))
    print("  Model:   ", config.get("model",    "not configured"))
    print_memory_report()


def run(path: str, provider: str = None, api_key: str = None,
        model: str = None, approve: bool = True) -> None:
    """
    Run the full MAIRA pipeline on an ML project.
    This is the one-liner entry point.

    Example:
        import maira
        maira.run("/path/to/my/ml/project")
    """
    print("\n  MAIRA — ML Agentic Intelligence for Research Automation")
    print("  " + "="*56)

    config      = get_or_setup_config()
    provider    = provider or config.get("provider", "groq")
    api_key     = api_key  or config.get("api_key",  None)
    model       = model    or config.get("model",    "llama-3.3-70b-versatile")
    ollama_host = config.get("ollama_host", "http://localhost:11434")

    print(f"  Provider: {provider} | Model: {model}\n")

    root         = os.path.abspath(path)
    project_scan = scan_project(root)
    parsed       = parse_all(project_scan.result_files, root)
    schema       = detect_schema(project_scan, parsed)

    updates = measure_outcomes(root)
    if updates:
        for u in updates:
            print(f"  [{u['outcome'].upper()}] {u['gap']} delta={u['delta']}")

    schema.experiment_gap = rank_gaps_by_history(schema.experiment_gap)
    print_memory_report()

    decisions = architect_dataset(project_scan.result_files, root)
    save_decisions(decisions, root)

    advice = get_advice(
        schema, decisions, parsed,
        api_key=api_key, provider=provider,
        model=model, ollama_host=ollama_host
    )
    save_advice(advice, root)
    save_report(root)

    written = run_hitl_pipeline(
        schema, project_scan, root,
        provider=provider, api_key=api_key,
        model=model, ollama_host=ollama_host
    )

    all_gaps      = schema.experiment_gap
    approved_gaps = all_gaps if written else []
    record_suggestions(all_gaps, approved_gaps, root)
    record_run(root, all_gaps, approved_gaps, provider)

    print("  MAIRA complete.\n")


__version__ = "0.1.0"
__all__     = ["scan", "advise", "generate", "configure", "status", "run"]