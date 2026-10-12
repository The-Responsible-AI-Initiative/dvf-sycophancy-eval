"""
Conversation runner for the dvf-sycophancy-eval pipeline (AIES 2026).

"Frictionless Validation: The Effects of AI Sycophancy on the Developing Mind"

Runs simulated conversations between age-calibrated child agents (Claude Haiku)
and three frontier target models to collect sycophancy scores across two
dimensions: emotional (S) and framing (F). Epistemic (E) is covered by SYCON
Bench and is excluded here.

Experimental design
-------------------
3 models × 3 age configs × 2 dimensions × 3 runs = 54 conversations total

Models : gpt55, claude_sonnet, gemini_flash
Ages   : a1 (7-9), a2 (10-12), a3 (13-17)
Dims   : S (emotional), F (framing/moral)
Runs   : 3 per cell

Usage
-----
python runner.py            # cost guard → pilot → full run
python runner.py --dry-run  # print plan without API calls

Environment variables required
-------------------------------
  ANTHROPIC_API_KEY  — agent (Haiku) + Claude Sonnet target
  OPENAI_API_KEY     — GPT-5.5 target
  GEMINI_API_KEY     — Gemini Flash target
"""

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

import agents

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TURNS         = 8
DIMENSIONS    = ["E", "F"]          # S (emotional) complete; this run covers E and F
AGE_CONFIGS   = ["a1", "a2", "a3"]
RUNS_PER_CELL = 3

AGENT_MODEL_HAIKU = "claude-haiku-4-5-20251001"   # used for F (framing)
AGENT_MODEL_LLAMA = "llama3.1:8b"                  # used for S (emotional) and E (epistemic)

TARGET_MODELS = {
    "gpt55": {
        "provider": "openai",
        "model_id": "gpt-5.4-mini",
    },
    "claude_sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-6",
    },
    "gemini_flash": {
        "provider": "google",
        "model_id": "gemini-2.5-flash",
    },
    "deepseek": {
        "provider": "deepseek",
        "model_id": "deepseek-chat",
    },
}

AGENT_MAP = {
    ("a1", "E"): "AGENT_A1_EPISTEMIC",
    ("a2", "E"): "AGENT_A2_EPISTEMIC",
    ("a3", "E"): "AGENT_A3_EPISTEMIC",
    ("a1", "S"): "AGENT_A1_EMOTIONAL",
    ("a2", "S"): "AGENT_A2_EMOTIONAL",
    ("a3", "S"): "AGENT_A3_EMOTIONAL",
    ("a1", "F"): "AGENT_A1_FRAMING",
    ("a2", "F"): "AGENT_A2_FRAMING",
    ("a3", "F"): "AGENT_A3_FRAMING",
}

RUN_MODELS = ["gpt55"]

_ROOT       = os.path.dirname(os.path.abspath(__file__))
RAW_DIR     = os.path.join(_ROOT, "results", "raw")
FAILED_DIR  = os.path.join(_ROOT, "results", "failed")
INVALID_DIR = os.path.join(_ROOT, "results", "invalid")

MAX_RETRIES = 3
RETRY_DELAY = 5.0
TOTAL       = len(RUN_MODELS) * len(AGE_CONFIGS) * len(DIMENSIONS) * RUNS_PER_CELL  # 27

# ---------------------------------------------------------------------------
# API call helpers
# ---------------------------------------------------------------------------

def call_llama_agent(system_prompt: str, conversation: list[dict]) -> str:
    import ollama
    messages = [{"role": "system", "content": system_prompt}]
    for msg in conversation:
        messages.append({"role": msg["role"], "content": msg["content"]})
    response = ollama.chat(
        model=AGENT_MODEL_LLAMA,
        messages=messages,
        options={"temperature": 1.0},
    )
    return response["message"]["content"]


def _call_anthropic_agent(system: str, messages: list[dict]) -> str:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=AGENT_MODEL_HAIKU,
        max_tokens=300,
        system=system,
        messages=messages,
        temperature=1.0,
    )
    return response.content[0].text


def _call_anthropic_target(model_id: str, messages: list[dict]) -> str:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model_id,
        max_tokens=4096,
        messages=messages,
        temperature=0,
    )
    return response.content[0].text


def _call_openai_target(model_id: str, messages: list[dict]) -> str:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_completion_tokens=300,
    )
    return response.choices[0].message.content


def _call_google_target(model_id: str, messages: list[dict]) -> str:
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    gemini_contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}],
        })

    response = client.models.generate_content(
        model=model_id,
        contents=gemini_contents,
        config={
            "temperature": 0,
            "max_output_tokens": 300,
        },
    )
    return response.text


def call_agent(system: str, messages: list[dict], dimension: str) -> str:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if dimension in ["S", "E"]:
                # Llama for emotional and epistemic
                return call_llama_agent(system, messages)
            else:
                return _call_anthropic_agent(system, messages)
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    model = AGENT_MODEL_LLAMA if dimension in ["S", "E"] else AGENT_MODEL_HAIKU
    raise RuntimeError(
        f"Agent ({model}) failed after {MAX_RETRIES} attempts: {last_err}"
    ) from last_err


def call_target(provider: str, model_id: str, messages: list[dict]) -> str:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if provider == "anthropic":
                return _call_anthropic_target(model_id, messages)
            if provider == "openai":
                return _call_openai_target(model_id, messages)
            if provider == "google":
                return _call_google_target(model_id, messages)
            raise ValueError(f"Unknown provider: {provider!r}")
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(
        f"Target ({provider}/{model_id}) failed after {MAX_RETRIES} attempts: {last_err}"
    ) from last_err


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

def _agent_perspective(conversation: list[dict]) -> list[dict]:
    """Reverse user/assistant roles so the agent sees itself as assistant."""
    role_map = {"user": "assistant", "assistant": "user"}
    return [{"role": role_map[m["role"]], "content": m["content"]} for m in conversation]


def run_conversation(
    model_key:  str,
    age_config: str,
    dim:        str,
    run_index:  int,
    verbose:    bool = False,
) -> list[dict]:
    """
    Runs one full conversation. Returns messages from the target's perspective.

    Args:
        model_key:  key into TARGET_MODELS
        age_config: one of a1 | a2 | a3
        dim:        one of S | F
        run_index:  0-indexed replication number
        verbose:    if True, print full turn content (used for pilot)
    """
    agent_id     = AGENT_MAP[(age_config, dim)]
    agent_prompt = getattr(agents, agent_id)
    target_cfg   = TARGET_MODELS[model_key]
    conversation = []

    for t in range(TURNS):
        agent_ctx = _agent_perspective(conversation)
        if not agent_ctx:
            agent_ctx = [{"role": "user", "content": "Begin the conversation."}]

        agent_msg = call_agent(agent_prompt, agent_ctx, dim)
        conversation.append({"role": "user", "content": agent_msg})

        target_msg = call_target(
            target_cfg["provider"],
            target_cfg["model_id"],
            conversation,
        )
        conversation.append({"role": "assistant", "content": target_msg})

        print(f"  Turn {t + 1}/{TURNS} complete")
        if verbose:
            print(f"\n  [Agent]\n  {agent_msg}\n")
            print(f"  [Target]\n  {target_msg}\n")

    return conversation


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _output_path(model: str, age: str, dim: str, run: int) -> str:
    return os.path.join(RAW_DIR, f"{model}_{age}_{dim}_{run}.json")


def _agent_model_for(dim: str) -> str:
    return AGENT_MODEL_LLAMA


def _save_conversation(
    model:    str,
    age:      str,
    dim:      str,
    run:      int,
    messages: list[dict],
) -> None:
    path = _output_path(model, age, dim, run)
    os.makedirs(RAW_DIR, exist_ok=True)
    payload = {
        "model":           model,
        "agent_id":        AGENT_MAP[(age, dim)],
        "age_config":      age,
        "scenario_type":   dim,
        "run_index":       run,
        "agent_model":     _agent_model_for(dim),
        "target_model_id": TARGET_MODELS[model]["model_id"],
        "num_turns":       TURNS,
        "created_at":      datetime.now(timezone.utc).isoformat(),
        "messages":        messages,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", dir=RAW_DIR, suffix=".tmp", delete=False
    ) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _save_failure(model: str, age: str, dim: str, run: int, error: str) -> None:
    os.makedirs(FAILED_DIR, exist_ok=True)
    path = os.path.join(FAILED_DIR, f"{model}_{age}_{dim}_{run}.json")
    payload = {
        "model":      model,
        "age_config": age,
        "dim":        dim,
        "run_index":  run,
        "error":      error,
        "failed_at":  datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Ollama check
# ---------------------------------------------------------------------------

def check_ollama() -> None:
    """
    Verifies that Ollama is running and llama3.1:8b is available.
    Exits with an error message if either check fails.
    """
    import subprocess
    result = subprocess.run(
        ["ollama", "list"], capture_output=True, text=True
    )
    print(result.stdout)
    if AGENT_MODEL_LLAMA not in result.stdout:
        print(
            f"ERROR: {AGENT_MODEL_LLAMA} not found in Ollama. "
            f"Run: ollama pull {AGENT_MODEL_LLAMA}"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _all_cells() -> list[tuple]:
    return [
        (model, age, dim, run)
        for model in RUN_MODELS
        for age   in AGE_CONFIGS
        for dim   in DIMENSIONS
        for run   in range(RUNS_PER_CELL)
    ]


def run_pilot(auto_confirm: bool = False) -> bool:
    """
    Runs one pilot conversation (claude_sonnet / a1 / S / run 0) with full
    turn-by-turn output. Saves immediately. Returns True if user approves
    launching the full run.
    """
    print("\n" + "=" * 60)
    print("PILOT: claude_sonnet | a1 | S | run 0")
    print("=" * 60 + "\n")

    path = _output_path("claude_sonnet", "a1", "S", 0)
    if os.path.exists(path):
        print(f"Pilot file already exists: {path}")
    else:
        messages = run_conversation("claude_sonnet", "a1", "S", 0, verbose=True)
        _save_conversation("claude_sonnet", "a1", "S", 0, messages)
        print(f"\nPilot saved to {path}")

    if auto_confirm:
        print("\nPilot complete. Auto-confirming full run (--yes).")
        return True
    print("\nPilot complete. Launch full run? (yes/no): ", end="", flush=True)
    return input().strip().lower() == "yes"


def run_all(dry_run: bool = False) -> None:
    """
    Runs all 54 (model, age, dim, run) cells, skipping any that already have
    an output file. Saves each conversation immediately after completion.
    Logs failures to results/failed/ and continues.
    """
    cells = _all_cells()

    if dry_run:
        print(f"\nDry run — {len(cells)} conversation(s) planned:\n")
        for model, age, dim, run in cells:
            print(f"  {_output_path(model, age, dim, run)}")
        print("\nDry run complete — no API calls made.")
        return

    n_done = 0
    n_skip = 0
    n_fail = 0

    for model, age, dim, run in cells:
        path  = _output_path(model, age, dim, run)
        label = f"{model} | {age} | {dim} | run {run + 1}/{RUNS_PER_CELL}"

        if os.path.exists(path):
            print(f"  Skipping (exists): {label}")
            n_skip += 1
            continue

        try:
            messages = run_conversation(model, age, dim, run)
            _save_conversation(model, age, dim, run, messages)
            n_done += 1
            print(f"[{n_done}/{TOTAL}] {label} | DVF pending scoring")
        except Exception:
            tb = traceback.format_exc()
            print(f"  FAILED: {label}\n{tb}")
            _save_failure(model, age, dim, run, tb)
            n_fail += 1

    print(
        f"\nDone. Completed: {n_done} | "
        f"Skipped: {n_skip} | Failed: {n_fail}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DVF sycophancy evaluation conversations (AIES 2026)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print run plan without making API calls.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip all confirmation prompts.",
    )
    args = parser.parse_args()

    print("\nEstimated cost (emotional S dimension, 27 conversations):")
    print("  GPT-5.5:       ~$4.54  (9 conversations)")
    print("  Claude Sonnet: ~$2.50  (9 conversations)")
    print("  Gemini Flash:  ~$0.10  (9 conversations)")
    print("  Agent (Llama): ~$0.00  (local Ollama)")
    print("  Total:         ~$7.14")

    if not args.dry_run and not args.yes:
        print("\nConfirm run? (yes/no): ", end="", flush=True)
        if input().strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(FAILED_DIR, exist_ok=True)
    os.makedirs(INVALID_DIR, exist_ok=True)

    if not args.dry_run:
        check_ollama()

    if args.dry_run:
        run_all(dry_run=True)
        return

    if not run_pilot(auto_confirm=args.yes):
        print("Full run aborted.")
        sys.exit(0)

    run_all()


if __name__ == "__main__":
    main()
