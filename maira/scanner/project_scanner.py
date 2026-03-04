"""
MAIRA — Project Scanner
Walks any ML project folder and extracts structure intelligence.
No code access needed — reads file names, folders, and result files only.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


RESULT_EXTENSIONS = {'.csv', '.json', '.npz', '.log', '.txt', '.yaml', '.yml'}
MODEL_EXTENSIONS  = {'.zip', '.pt', '.pth', '.h5', '.pkl', '.joblib', '.ckpt'}
CODE_EXTENSIONS   = {'.py', '.ipynb', '.r', '.R'}

SKIP_DIRS  = {'.git', '__pycache__', 'venv', '.venv', 'node_modules', '.pytest_cache'}
SKIP_FILES = {
    'maira_dataset_decisions.json',
    'maira_research_advice.json',
    'maira_dependency_graph.json',
    'maira_hyperparam_map.json',
    'maira_reproducibilty.json',
    'maira_detections.json',
    'maira_report.md',
    'requirements.txt',
}

@dataclass
class ProjectScan:
    root_path:       str
    all_folders:     List[str] = field(default_factory=list)
    result_files:    List[str] = field(default_factory=list)
    model_files:     List[str] = field(default_factory=list)
    code_files:      List[str] = field(default_factory=list)
    experiment_dirs: List[str] = field(default_factory=list)


def scan_project(root_path: str) -> ProjectScan:
    """Walk project folder and collect all relevant paths."""
    root = Path(root_path).resolve()
    scan = ProjectScan(root_path=str(root))

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        rel_dir = Path(dirpath).relative_to(root)
        if str(rel_dir) != '.':
            scan.all_folders.append(str(rel_dir))

        for fname in filenames:
            full = Path(dirpath) / fname
            rel  = str(full.relative_to(root))

            if fname in SKIP_FILES or rel.startswith("maira/"):
                continue
            ext  = full.suffix.lower()

            if ext in RESULT_EXTENSIONS:
                scan.result_files.append(rel)
            if ext in MODEL_EXTENSIONS:
                scan.model_files.append(rel)
            if ext in CODE_EXTENSIONS:
                scan.code_files.append(rel)

    EXPERIMENT_KEYWORDS = {
        'baseline', 'experiment', 'run', 'trial', 'latency',
        'partial', 'obs', 'degraded', 'ensemble', 'ablation',
        'sweep', 'ppo', 'sac', 'dqn', 'a2c', 'ddpg', 'td3'
    }
    for folder in scan.all_folders:
        parts = set(Path(folder).parts)
        if parts & EXPERIMENT_KEYWORDS:
            scan.experiment_dirs.append(folder)

    return scan


def print_scan(scan: ProjectScan) -> None:
    print("\n" + "="*60)
    print("  MAIRA — Project Scan")
    print("="*60)
    print(f"\nRoot: {scan.root_path}")
    print(f"Total folders:      {len(scan.all_folders)}")
    print(f"Result files:       {len(scan.result_files)}")
    print(f"Model files:        {len(scan.model_files)}")
    print(f"Code files:         {len(scan.code_files)}")
    print(f"Experiment dirs:    {len(scan.experiment_dirs)}")

    if scan.result_files:
        print("\nResult files found:")
        for f in scan.result_files:
            print(f"  {f}")

    if scan.model_files:
        print("\nModel files found:")
        for f in scan.model_files:
            print(f"  {f}")

    if scan.experiment_dirs:
        print("\nExperiment variants detected:")
        for d in scan.experiment_dirs:
            print(f"  {d}")

    print()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    scan = scan_project(path)
    print_scan(scan)