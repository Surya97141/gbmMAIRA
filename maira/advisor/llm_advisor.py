"""
MAIRA — LLM Research Advisor
Reads all scan + dataset results and gives AI-powered recommendations.
Supports Groq (free), Ollama (local), Gemini, and Anthropic.
User brings their own key — MAIRA never stores it permanently.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class ResearchAdvice:
    project_type:          str
    gaps_found:            List[str]
    diagnosis:             str
    top3_next_experiments: List[str]
    raw_response:          str


def _build_prompt(schema, decisions, parsed) -> str:
    gaps = "\n".join(f"  - {g}" for g in schema.experiment_gap)

    dataset_summary = []
    for d in decisions:
        dataset_summary.append(
            f"  {d.file_path}: {d.total_records} records, "
            f"{d.dataset_type}, split {d.split_ratio} "
            f"(train={d.train_size}/val={d.val_size}/test={d.test_size})"
        )
    datasets = "\n".join(dataset_summary)

    result_summary = []
    for r in parsed:
        if not r.error and r.file_type in ('json', 'npz', 'csv'):
            result_summary.append(f"  {r.file_path}: {r.summary}")
    results = "\n".join(result_summary)

    return f"""You are an expert ML research advisor analyzing a {schema.project_type} project.

PROJECT SCAN SUMMARY:
- Project type: {schema.project_type} (confidence: {schema.confidence})
- Evidence: {', '.join(schema.evidence)}

EXPERIMENT GAPS DETECTED:
{gaps if gaps else '  None detected'}

DATASETS FOUND AND SPLIT:
{datasets}

RESULT FILE CONTENTS:
{results}

Based on this project history and gaps detected, provide:

1. DIAGNOSIS (2-3 sentences): What is the current state of this research?

2. TOP 3 NEXT EXPERIMENTS (ranked by expected impact):
   - Experiment 1: [name] — [what to run] — [why this gap matters]
   - Experiment 2: [name] — [what to run] — [why this gap matters]
   - Experiment 3: [name] — [what to run] — [why this gap matters]

3. REWARD/OBSERVATION FIX: One specific code-level change to improve performance.

Be specific. Reference actual file names, gap names, and numbers from the data."""


def _call_groq(prompt: str, api_key: str,
               model: str = "llama3-8b-8192") -> str:
    from groq import Groq
    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model      = model,
        messages   = [{"role": "user", "content": prompt}],
        max_tokens = 1500
    )
    return response.choices[0].message.content


def _call_ollama(prompt: str, model: str = "llama3.2",
                 host: str = "http://localhost:11434") -> str:
    import requests
    response = requests.post(
        f"{host}/api/generate",
        json   = {"model": model, "prompt": prompt, "stream": False},
        timeout = 300
    )
    data = response.json()
    if "response" not in data:
        raise RuntimeError(f"Ollama error: {data}")
    return data["response"]


def _call_gemini(prompt: str, api_key: str) -> str:
    from google import genai
    client   = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model    = "gemini-2.0-flash",
        contents = prompt
    )
    return response.text


def _call_anthropic(prompt: str, api_key: str) -> str:
    import anthropic
    client  = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model      = "claude-sonnet-4-6",
        max_tokens = 1500,
        messages   = [{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def get_advice(schema, decisions, parsed,
               api_key:     str = None,
               provider:    str = "groq",
               model:       str = "llama3-8b-8192",
               ollama_host: str = "http://localhost:11434") -> ResearchAdvice:

    prompt = _build_prompt(schema, decisions, parsed)
    print(f"  Sending to {provider} ({model}) for research advice...\n")

    if provider == "groq":
        raw = _call_groq(prompt, api_key, model)
    elif provider == "ollama":
        raw = _call_ollama(prompt, model, ollama_host)
    elif provider == "gemini":
        raw = _call_gemini(prompt, api_key)
    elif provider == "anthropic":
        raw = _call_anthropic(prompt, api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return ResearchAdvice(
        project_type          = schema.project_type,
        gaps_found            = schema.experiment_gap,
        diagnosis             = "",
        top3_next_experiments = [],
        raw_response          = raw
    )


def print_advice(advice: ResearchAdvice) -> None:
    print("\n" + "="*60)
    print("  MAIRA — Research Advisor")
    print("="*60)
    print(advice.raw_response)
    print()


def save_advice(advice: ResearchAdvice, root: str) -> None:
    out = {
        "project_type": advice.project_type,
        "gaps_found":   advice.gaps_found,
        "advice":       advice.raw_response,
    }
    out_path = Path(root) / "maira_research_advice.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Advice saved to {out_path}")