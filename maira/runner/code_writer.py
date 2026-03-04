"""
MAIRA — Code Writer Agent with Human-in-the-Loop Approval Gate
Reads gaps → shows possibilities → waits for human approval → writes code only if approved.
NEVER modifies original files — all output goes to maira/generated/
NEVER writes code without explicit human approval.
"""

import os
import json
from pathlib import Path
from datetime import datetime


def _build_preview_prompt(gap: str, project_type: str) -> str:
    return f"""You are an ML research advisor.

PROJECT TYPE: {project_type}
PROPOSED EXPERIMENT: {gap}

Reply in exactly 4 lines, no headers, no bullets:
LINE 1 - What this experiment runs (one sentence)
LINE 2 - Expected impact: High / Medium / Low and one reason why
LINE 3 - Estimated training time (e.g. 30 mins on GPU)
LINE 4 - Risk: one thing that could go wrong

Just 4 plain lines. Nothing else."""


def _build_code_prompt(gap: str, project_type: str) -> str:
    return f"""You are an ML engineer. Write a complete runnable Python experiment script.

PROJECT TYPE: {project_type}
EXPERIMENT: {gap}

EXISTING PROJECT:
- environment/base_env.py contains AerialCombatEnv (gym-compatible)
- observation space: 12 values (agent_pos/vel + opponent_pos/vel)
- action space: 4 continuous [-1, 1] (throttle, pitch, roll, yaw)
- train_ppo.py is the existing PPO training script

STRICT RULES:
1. Import AerialCombatEnv from environment.base_env
2. Use stable_baselines3 only — no other RL libraries
3. Save model to models/ with descriptive name
4. Save results to logs/metrics/ as CSV with columns: timestep, mean_reward, std_reward
5. Print progress every 10000 steps
6. Add comment at top explaining what gap this fills
7. Under 80 lines total
8. Write ONLY Python code — no explanation, no markdown fences

Start with the comment, then imports, then code."""


def _call_groq(prompt: str, api_key: str,
               model: str = "llama3-8b-8192") -> str:
    from groq import Groq
    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model      = model,
        messages   = [{"role": "user", "content": prompt}],
        max_tokens = 2000
    )
    return response.choices[0].message.content


def _call_ollama(prompt: str, model: str = "llama3.2",
                 host: str = "http://localhost:11434") -> str:
    import requests
    response = requests.post(
        f"{host}/api/generate",
        json    = {"model": model, "prompt": prompt, "stream": False},
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
        max_tokens = 2000,
        messages   = [{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def _llm_call(prompt: str, provider: str, api_key: str = None,
              model: str = "llama3-8b-8192",
              ollama_host: str = "http://localhost:11434") -> str:
    if provider == "groq":
        return _call_groq(prompt, api_key, model)
    elif provider == "ollama":
        return _call_ollama(prompt, model, ollama_host)
    elif provider == "gemini":
        return _call_gemini(prompt, api_key)
    elif provider == "anthropic":
        return _call_anthropic(prompt, api_key)
    raise ValueError(f"Unknown provider: {provider}")


def _clean_code(raw: str) -> str:
    """Strip markdown fences if LLM adds them."""
    lines    = raw.strip().splitlines()
    clean    = []
    in_block = False
    for line in lines:
        if line.strip().startswith("```"):
            in_block = not in_block
            continue
        clean.append(line)
    return "\n".join(clean).strip()


def _save_script(code: str, gap: str, output_dir: str) -> str:
    safe_name = gap.lower()
    for ch in [' ', '—', '-', '/', '\\', ':', '.']:
        safe_name = safe_name.replace(ch, '_')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
    safe_name = safe_name[:60]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"exp_{safe_name}_{timestamp}.py"
    out_path  = Path(output_dir) / filename
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(code)
    return str(out_path)


def show_possibilities(schema, provider: str, api_key: str = None,
                       model: str = "llama3-8b-8192",
                       ollama_host: str = "http://localhost:11434") -> list:
    print("\n" + "="*60)
    print("  MAIRA — Experiment Possibilities")
    print("="*60)
    print(f"\n  Found {len(schema.experiment_gap)} experiment gaps.")
    print("  Generating previews...\n")

    possibilities = []

    for i, gap in enumerate(schema.experiment_gap):
        print(f"  [{i+1}] {gap}")
        print("       Analyzing...")

        preview = _llm_call(
            _build_preview_prompt(gap, schema.project_type),
            provider, api_key, model, ollama_host
        )

        lines = [l.strip() for l in preview.strip().splitlines() if l.strip()]

        print(f"       What:   {lines[0] if len(lines) > 0 else 'N/A'}")
        print(f"       Impact: {lines[1] if len(lines) > 1 else 'N/A'}")
        print(f"       Time:   {lines[2] if len(lines) > 2 else 'N/A'}")
        print(f"       Risk:   {lines[3] if len(lines) > 3 else 'N/A'}")
        print()

        possibilities.append((gap, preview))

    return possibilities


def approval_gate(possibilities: list) -> list:
    print("="*60)
    print("  MAIRA — Approval Gate")
    print("="*60)
    print("\n  Review the experiments above.")
    print(f"  Type numbers to approve — valid range: 1 to {len(possibilities)}")
    print(f"  (e.g. '1' or '1,2' or 'all' or 'none'):")
    print()

    while True:
        user_input = input("  Your choice: ").strip().lower()

        if user_input == "none":
            print("\n  No experiments approved. No code will be written.")
            return []

        if user_input == "all":
            approved = [g for g, _ in possibilities]
            print(f"\n  All {len(approved)} experiments approved.")
            return approved

        try:
            indices  = [int(x.strip()) - 1 for x in user_input.split(",")]
            approved = []
            valid    = True
            for idx in indices:
                if 0 <= idx < len(possibilities):
                    approved.append(possibilities[idx][0])
                else:
                    print(f"  Invalid number: {idx+1}. Try again.")
                    valid = False
                    break
            if valid and approved:
                print(f"\n  {len(approved)} experiment(s) approved:")
                for g in approved:
                    print(f"    - {g}")
                return approved
        except ValueError:
            print("  Invalid input. Type numbers like: 1,2 or 'all' or 'none'")


def write_approved(approved: list, schema, project_root: str,
                   provider: str, api_key: str = None,
                   model: str = "llama3-8b-8192",
                   ollama_host: str = "http://localhost:11434") -> list:
    if not approved:
        return []

    output_dir = os.path.join(project_root, "maira", "generated")
    written    = []

    print("\n" + "="*60)
    print("  MAIRA — Code Writer")
    print("="*60)
    print(f"  Writing {len(approved)} approved experiment(s)...")
    print(f"  Output: {output_dir}")
    print("  Original files: NEVER modified\n")

    for gap in approved:
        print(f"  Writing: {gap}")
        raw  = _llm_call(
            _build_code_prompt(gap, schema.project_type),
            provider, api_key, model, ollama_host
        )
        code = _clean_code(raw)
        path = _save_script(code, gap, output_dir)
        written.append(path)
        print(f"  Saved:   {path}\n")

    print("="*60)
    print("  MAIRA — Final Review Gate")
    print("="*60)
    print(f"\n  {len(written)} script(s) written to maira/generated/")
    print("  Review them in VS Code before running.")
    print("\n  Files written:")
    for w in written:
        print(f"    {w}")
    print()
    print("  To run: python <script_path>")
    print("  Original files: untouched.")

    return written


def run_hitl_pipeline(schema, project_scan, project_root: str,
                      provider:    str = "groq",
                      api_key:     str = None,
                      model:       str = "llama3-8b-8192",
                      ollama_host: str = "http://localhost:11434") -> list:

    possibilities = show_possibilities(
        schema, provider, api_key, model, ollama_host
    )
    if not possibilities:
        print("  No gaps found. Nothing to generate.")
        return []

    approved = approval_gate(possibilities)
    written  = write_approved(
        approved, schema, project_root,
        provider, api_key, model, ollama_host
    )
    return written