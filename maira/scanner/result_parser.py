"""
MAIRA — Result Parser
Reads every result file found by the scanner and extracts metrics.
Works on CSV, JSON, NPZ — no assumptions about format.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ParsedResult:
    file_path:   str
    file_type:   str
    summary:     Dict[str, Any] = field(default_factory=dict)
    error:       Optional[str]  = None


def parse_csv(path: Path) -> ParsedResult:
    try:
        df = pd.read_csv(path)
        summary = {
            "rows":    len(df),
            "columns": list(df.columns),
            "numeric_summary": df.describe().to_dict() if not df.empty else {},
        }
        return ParsedResult(file_path=str(path), file_type="csv", summary=summary)
    except Exception as e:
        return ParsedResult(file_path=str(path), file_type="csv", error=str(e))


def parse_json(path: Path) -> ParsedResult:
    try:
        with open(path) as f:
            data = json.load(f)

        if isinstance(data, list):
            summary = {
                "records": len(data),
                "keys":    list(data[0].keys()) if data else [],
            }
        elif isinstance(data, dict):
            summary = {
                "keys":    list(data.keys()),
                "records": 1,
            }
        else:
            summary = {"type": type(data).__name__}

        return ParsedResult(file_path=str(path), file_type="json", summary=summary)
    except Exception as e:
        return ParsedResult(file_path=str(path), file_type="json", error=str(e))


def parse_npz(path: Path) -> ParsedResult:
    try:
        data = np.load(path, allow_pickle=True)
        summary = {
            "arrays": list(data.files),
            "shapes": {k: list(data[k].shape) for k in data.files},
        }
        return ParsedResult(file_path=str(path), file_type="npz", summary=summary)
    except Exception as e:
        return ParsedResult(file_path=str(path), file_type="npz", error=str(e))


def parse_file(file_path: str, root: str) -> ParsedResult:
    """Parse a single result file based on its extension."""
    full = Path(root) / file_path
    ext  = full.suffix.lower()

    if ext == ".csv":
        return parse_csv(full)
    elif ext == ".json":
        return parse_json(full)
    elif ext == ".npz":
        return parse_npz(full)
    else:
        return ParsedResult(file_path=file_path, file_type=ext,
                            summary={"note": "format not parsed"})


def parse_all(result_files: List[str], root: str) -> List[ParsedResult]:
    """Parse all result files from a project scan."""
    results = []
    for f in result_files:
        parsed = parse_file(f, root)
        results.append(parsed)
    return results


def print_parsed(results: List[ParsedResult]) -> None:
    print("\n" + "="*60)
    print("  MAIRA — Result Parser")
    print("="*60)
    for r in results:
        print(f"\n  {r.file_path}  [{r.file_type}]")
        if r.error:
            print(f"    ERROR: {r.error}")
        else:
            for k, v in r.summary.items():
                if k == "numeric_summary":
                    continue
                print(f"    {k}: {v}")
    print()


if __name__ == "__main__":
    import sys
    from project_scanner import scan_project

    root = sys.argv[1] if len(sys.argv) > 1 else "."
    scan = scan_project(root)
    results = parse_all(scan.result_files, root)
    print_parsed(results)