"""
Targeted rerun: deepseek a1 F run_index 0 and 1.
Agent: claude-haiku-4-5-20251001
Target: deepseek-v4-pro via api.deepseek.com
"""

import json
import os
import tempfile
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

import agents

TURNS           = 8
AGENT_MODEL     = "claude-haiku-4-5-20251001"
TARGET_MODEL_ID = "deepseek-v4-pro"
TARGET_BASE_URL = "https://api.deepseek.com"
RAW_DIR         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "raw")
MAX_RETRIES     = 3
RETRY_DELAY     = 5.0

RERUNS = [
    {"model": "deepseek", "age": "a1", "dim": "F", "run_index": 0},
    {"model": "deepseek", "age": "a1", "dim": "F", "run_index": 1},
]


def call_haiku_agent(system: str, messages: list[dict]) -> str:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=AGENT_MODEL,
        max_tokens=300,
        system=system,
        messages=messages,
        temperature=1.0,
    )
    return response.content[0].text


def call_deepseek_target(messages: list[dict]) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=TARGET_BASE_URL,
    )
    response = client.chat.completions.create(
        model=TARGET_MODEL_ID,
        messages=messages,
        temperature=0,
        max_tokens=1000,
    )
    content = response.choices[0].message.content
    if not content and hasattr(response.choices[0].message, "reasoning_content"):
        content = getattr(response.choices[0].message, "reasoning_content")
    return content


def call_agent(system: str, messages: list[dict]) -> str:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_haiku_agent(system, messages)
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"Agent failed after {MAX_RETRIES} attempts: {last_err}") from last_err


def call_target(messages: list[dict]) -> str:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_deepseek_target(messages)
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"Target failed after {MAX_RETRIES} attempts: {last_err}") from last_err


def agent_perspective(conversation: list[dict]) -> list[dict]:
    role_map = {"user": "assistant", "assistant": "user"}
    return [{"role": role_map[m["role"]], "content": m["content"]} for m in conversation]


def run_conversation(age: str, dim: str) -> list[dict]:
    dim_label    = "EMOTIONAL" if dim == "S" else "FRAMING"
    agent_id     = f"AGENT_{age.upper()}_{dim_label}"
    agent_prompt = getattr(agents, agent_id)
    conversation = []

    for t in range(TURNS):
        p_ctx = agent_perspective(conversation)
        agent_ctx = [{"role": "user", "content": "Begin the conversation."}] + p_ctx
        
        agent_msg = call_agent(agent_prompt, agent_ctx)
        conversation.append({"role": "user", "content": agent_msg})

        target_msg = call_target(conversation)
        if not target_msg or not target_msg.strip():
            target_msg = "[No response]"
        
        conversation.append({"role": "assistant", "content": target_msg})
        print(f"  Turn {t + 1}/{TURNS} complete")

    return conversation


def save_conversation(model: str, age: str, dim: str, run: int, messages: list[dict]) -> None:
    path = os.path.join(RAW_DIR, f"{model}_{age}_{dim}_{run}.json")
    dim_label = "EMOTIONAL" if dim == "S" else "FRAMING"
    agent_id = f"AGENT_{age.upper()}_{dim_label}"
    payload = {
        "model":           model,
        "agent_id":        agent_id,
        "age_config":      age,
        "scenario_type":   dim,
        "run_index":       run,
        "agent_model":     AGENT_MODEL,
        "target_model_id": TARGET_MODEL_ID,
        "num_turns":       TURNS,
        "created_at":      datetime.now(timezone.utc).isoformat(),
        "messages":        messages,
    }
    os.makedirs(RAW_DIR, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=RAW_DIR, suffix=".tmp", delete=False) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    for r in RERUNS:
        label = f"{r['model']} | {r['age']} | {r['dim']} | run {r['run_index']}"
        print(f"\nRunning: {label}")
        messages = run_conversation(r["age"], r["dim"])
        save_conversation(r["model"], r["age"], r["dim"], r["run_index"], messages)
        print(f"Done: {label}")

    print("\nAll reruns complete.")
