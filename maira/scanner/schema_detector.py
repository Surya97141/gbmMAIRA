"""
MAIRA — Schema Detector
Looks at scan + parsed results and classifies the ML project type.
No code access needed — uses folder names, file names, and data keys only.
"""

from dataclasses import dataclass, field
from typing import List, Dict
from project_scanner import ProjectScan
from result_parser import ParsedResult


RL_KEYWORDS    = {'ppo','sac','dqn','a2c','ddpg','td3','maddpg','episode','reward',
                  'timestep','policy','env','action','observation','latency','combat',
                  'aerial','agent','opponent','replay'}
CV_KEYWORDS    = {'image','img','pixel','conv','resnet','yolo','detection','segmentation',
                  'classification','augment','bbox','cnn'}
NLP_KEYWORDS   = {'token','embed','transformer','bert','gpt','llm','text','vocab',
                  'sentence','language','corpus','nlp'}
TABULAR_KEYWORDS = {'feature','label','target','train_test','split','accuracy',
                    'precision','recall','f1','xgb','lgbm','sklearn'}


@dataclass
class ProjectSchema:
    project_type:   str
    confidence:     str
    evidence:       List[str] = field(default_factory=list)
    experiment_gap: List[str] = field(default_factory=list)


def _collect_tokens(scan: ProjectScan, results: List[ParsedResult]) -> set:
    """Collect all meaningful tokens from folder names and result keys."""
    tokens = set()

    for folder in scan.all_folders:
        for part in folder.replace('/', ' ').replace('_', ' ').split():
            tokens.add(part.lower())

    for r in results:
        if not r.error:
            for k in r.summary.get('keys', []):
                for part in k.replace('_', ' ').split():
                    tokens.add(part.lower())
            for arr in r.summary.get('arrays', []):
                tokens.add(arr.lower())

    return tokens


def _detect_gaps(scan: ProjectScan) -> List[str]:
    """Find unexplored experiment combinations."""
    gaps = []
    exp_dirs = [d.lower() for d in scan.experiment_dirs]
    joined   = ' '.join(exp_dirs)

    # Latency gaps
    if 'latency_10ms'  in joined and \
       'latency_50ms'  in joined and \
       'latency_100ms' in joined and \
       'latency_25ms'  not in joined:
        gaps.append("latency_25ms never tested — gap between 10ms and 50ms")

    if 'latency_100ms' in joined and 'latency_200ms' not in joined:
        gaps.append("latency_200ms never tested — upper bound unknown")

    # Observation gaps
    if 'obs_20' in joined and 'obs_50' in joined and 'obs_80' in joined:
        if 'obs_30' not in joined:
            gaps.append("obs_30_percent never tested — gap in observation sweep")
        if 'obs_10' not in joined:
            gaps.append("obs_10_percent never tested — extreme degradation unknown")

    # Combined degradation gaps
    if 'degraded_combined' in joined:
        if 'obs_20_latency' not in joined:
            gaps.append("obs_20 + latency_100ms combined never tested")
        if 'obs_80_latency' not in joined:
            gaps.append("obs_80 + latency_10ms combined never tested")

    # Algorithm gaps
    if 'ppo' in joined and 'sac' not in joined:
        gaps.append("SAC never benchmarked — only PPO tested")
    if 'ppo' in joined and 'td3' not in joined:
        gaps.append("TD3 never benchmarked against PPO baseline")

    return gaps


def detect_schema(scan: ProjectScan,
                  results: List[ParsedResult]) -> ProjectSchema:
    tokens = _collect_tokens(scan, results)

    scores = {
        "REINFORCEMENT_LEARNING": len(tokens & RL_KEYWORDS),
        "COMPUTER_VISION":        len(tokens & CV_KEYWORDS),
        "NLP":                    len(tokens & NLP_KEYWORDS),
        "TABULAR_ML":             len(tokens & TABULAR_KEYWORDS),
    }

    best_type  = max(scores, key=scores.get)
    best_score = scores[best_type]

    if best_score >= 5:
        confidence = "HIGH"
    elif best_score >= 2:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
        best_type  = "UNKNOWN"

    # Collect evidence
    evidence = []
    for keyword_set, label in [
        (RL_KEYWORDS, "RL"), (CV_KEYWORDS, "CV"),
        (NLP_KEYWORDS, "NLP"), (TABULAR_KEYWORDS, "Tabular")
    ]:
        matched = tokens & keyword_set
        if matched:
            evidence.append(f"{label} keywords matched: {', '.join(sorted(matched)[:5])}")

    gaps = _detect_gaps(scan)

    return ProjectSchema(
        project_type=best_type,
        confidence=confidence,
        evidence=evidence,
        experiment_gap=gaps
    )


def print_schema(schema: ProjectSchema) -> None:
    print("\n" + "="*60)
    print("  MAIRA — Schema Detector")
    print("="*60)
    print(f"\n  Project type:  {schema.project_type}")
    print(f"  Confidence:    {schema.confidence}")

    print("\n  Evidence:")
    for e in schema.evidence:
        print(f"    {e}")

    if schema.experiment_gap:
        print("\n  Experiment gaps detected:")
        for g in schema.experiment_gap:
            print(f"    GAP: {g}")
    else:
        print("\n  No gaps detected.")
    print()


if __name__ == "__main__":
    import sys
    from project_scanner import scan_project, print_scan
    from result_parser   import parse_all

    root    = sys.argv[1] if len(sys.argv) > 1 else "."
    scan    = scan_project(root)
    results = parse_all(scan.result_files, root)
    schema  = detect_schema(scan, results)
    print_schema(schema)