"""
MAIRA — First Run Setup Wizard
Asks user which LLM provider to use.
Saves choice to maira/.config.json
Never asks again after first run.
"""

import json
import os
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / ".config.json"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def save_config(config: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def test_groq(api_key: str) -> bool:
    try:
        from groq import Groq
        client   = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model      = "llama-3.3-70b-versatile",
            messages   = [{"role": "user", "content": "Say OK"}],
            max_tokens = 5
        )
        return True
    except Exception as e:
        print(f"  Groq test failed: {e}")
        return False


def test_ollama(host: str = "http://localhost:11434") -> bool:
    try:
        import requests
        r = requests.get(f"{host}/api/tags", timeout=5)
        return r.status_code == 200
    except:
        return False


def test_gemini(api_key: str) -> bool:
    try:
        from google import genai
        client   = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model    = "gemini-2.0-flash",
            contents = "Say OK"
        )
        return True
    except Exception as e:
        print(f"  Gemini test failed: {e}")
        return False


def test_anthropic(api_key: str) -> bool:
    try:
        import anthropic
        client  = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 5,
            messages   = [{"role": "user", "content": "Say OK"}]
        )
        return True
    except Exception as e:
        print(f"  Anthropic test failed: {e}")
        return False


def run_wizard() -> dict:
    print("\n" + "="*60)
    print("  MAIRA — First Run Setup")
    print("="*60)
    print("\n  Welcome to MAIRA!")
    print("  Let's set up your LLM provider.\n")
    print("  Options:")
    print("  [1] Groq      — free tier, no GPU needed, no credit card")
    print("                  14,400 requests/day limit")
    print("                  Get key: console.groq.com")
    print("  [2] Ollama    — local, no internet, no key needed")
    print("                  needs Ollama installed + model pulled")
    print("                  GPU/CPU dependent — model must fit in RAM")
    print("  [3] Gemini    — free tier, Google account needed")
    print("                  low daily quota, may exhaust quickly")
    print("                  Get key: aistudio.google.com")
    print("  [4] Anthropic — paid only, no free tier")
    print("                  Get key: console.anthropic.com")
    print()

    while True:
        choice = input("  Your choice [1-4]: ").strip()
        if choice in ["1", "2", "3", "4"]:
            break
        print("  Please enter 1, 2, 3, or 4.")

    config = {}

    # ── Groq ──────────────────────────────────────────
    if choice == "1":
        print("\n  Get your free Groq key at: console.groq.com")
        print("  Sign in → API Keys → Create key → Copy it\n")
        api_key = input("  Paste your Groq API key: ").strip()
        print("  Testing connection...")
        if test_groq(api_key):
            print("  Groq connected successfully!")
        else:
            print("  Connection failed. Saving anyway — check key later.")
        config = {
            "provider": "groq",
            "api_key":  api_key,
            "model":    "llama-3.3-70b-versatile"
        }

    # ── Ollama ────────────────────────────────────────
    elif choice == "2":
        print("\n  Checking for Ollama...")

        wsl_host     = _get_wsl_host()
        hosts_to_try = [
            wsl_host,
            "http://localhost:11434",
            "http://127.0.0.1:11434",
        ]

        ollama_host = None
        for host in hosts_to_try:
            if host and test_ollama(host):
                ollama_host = host
                break

        if not ollama_host:
            print("  Ollama not found. Make sure it is running.")
            print("  Windows: open PowerShell →")
            print("    $env:OLLAMA_HOST='0.0.0.0:11434'")
            print("    ollama serve")
            ollama_host = input(
                "  Enter Ollama host (press Enter for default): "
            ).strip()
            if not ollama_host:
                ollama_host = "http://localhost:11434"

        print(f"\n  Ollama host: {ollama_host}")
        print("\n  Recommended models for RTX 4050 / 16GB RAM:")
        print("  [1] llama3.2:3b    — 2GB VRAM — fits comfortably")
        print("  [2] phi3:mini      — 2.3GB VRAM — fits comfortably")
        print("  [3] mistral:7b-q4  — 4.1GB VRAM — fits")
        print("  [4] deepseek-coder — 1GB VRAM — already installed")
        print()
        print("  To pull a model: ollama pull llama3.2:3b")

        model_choice = input(
            "\n  Pick model [1-4] or type custom name: "
        ).strip()
        model_map = {
            "1": "llama3.2:3b",
            "2": "phi3:mini",
            "3": "mistral:7b-q4",
            "4": "deepseek-coder"
        }
        model = model_map.get(
            model_choice,
            model_choice if model_choice else "llama3.2:3b"
        )

        config = {
            "provider":    "ollama",
            "api_key":     None,
            "model":       model,
            "ollama_host": ollama_host
        }
        print(f"  Ollama configured — model: {model}")

    # ── Gemini ────────────────────────────────────────
    elif choice == "3":
        print("\n  Get your free Gemini key at: aistudio.google.com")
        api_key = input("  Paste your Gemini API key: ").strip()
        print("  Testing connection...")
        if test_gemini(api_key):
            print("  Gemini connected successfully!")
        else:
            print("  Connection failed. Saving anyway — check key later.")
        config = {
            "provider": "gemini",
            "api_key":  api_key,
            "model":    "gemini-2.0-flash"
        }

    # ── Anthropic ─────────────────────────────────────
    elif choice == "4":
        print("\n  Get your Anthropic key at: console.anthropic.com")
        api_key = input("  Paste your Anthropic API key: ").strip()
        print("  Testing connection...")
        if test_anthropic(api_key):
            print("  Anthropic connected successfully!")
        else:
            print("  Connection failed. Saving anyway — check key later.")
        config = {
            "provider": "anthropic",
            "api_key":  api_key,
            "model":    "claude-sonnet-4-6"
        }

    save_config(config)
    print(f"\n  Config saved to {CONFIG_PATH}")
    print("  MAIRA will not ask again.")
    print("  To change provider: delete maira/.config.json and rerun.\n")

    return config


def _get_wsl_host() -> str:
    """Get Windows host IP from WSL."""
    try:
        import subprocess
        result = subprocess.run(
            ["ip", "route"],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if "default" in line:
                return "http://" + line.split()[2] + ":11434"
    except:
        pass
    return None


def get_or_setup_config() -> dict:
    """Load config if exists, show it and ask to keep or change."""
    config = load_config()

    if config:
        print("\n" + "="*60)
        print("  MAIRA — Provider Setup")
        print("="*60)
        print(f"\n  Current provider: {config['provider']}")
        print(f"  Current model:    {config['model']}")
        print()
        choice = input("  Keep this? [Y/n]: ").strip().lower()
        if choice in ["", "y", "yes"]:
            return config
        print()

    return run_wizard()

if __name__ == "__main__":
    config = run_wizard()
    print(f"  Provider: {config['provider']}")
    print(f"  Model:    {config['model']}")