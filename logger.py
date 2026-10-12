"""
Run logger for the dvf-sycophancy-eval pipeline.

Tracks conversation runs, scoring status, and DVF computation progress.
All state is persisted atomically to data/run_log.json.

Schema (data/run_log.json)
--------------------------
{
  "version": "1.0",
  "runs": [
    {
      "run_id":           str,   # uuid4
      "agent_id":         str,   # e.g. AGENT_A1_EPISTEMIC
      "model":            str,   # gpt4o | claude_sonnet | gemini_pro
      "age_config":       str,   # a1 | a2 | a3
      "scenario_type":    str,   # E | S | F
      "status":           str,   # pending | running | completed |
                                 # scored | dvf_computed | failed
      "conversation_path": str | null,
      "created_at":       str,   # ISO 8601 UTC
      "completed_at":     str | null,
      "scored_at":        str | null,
      "dvf_computed_at":  str | null,
      "error":            str | null
    }
  ]
}

Usage
-----
    import logger

    # Register a run before starting
    run_id = logger.register(
        agent_id="AGENT_A1_EPISTEMIC",
        model="gpt4o",
        age_config="a1",
        scenario_type="E",
    )

    # Update status at each stage
    logger.mark_running(run_id)
    logger.mark_completed(run_id, conversation_path="data/conversations/...")
    logger.mark_scored(run_id)
    logger.mark_dvf_computed(run_id)

    # Mark failed on error
    logger.mark_failed(run_id, error=str(e))

    # Query
    pending = logger.get_by_status("pending")
    summary = logger.summary()
"""

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone

_ROOT    = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(_ROOT, "data", "run_log.json")

VALID_STATUSES = frozenset({
    "pending", "running", "completed", "scored", "dvf_computed", "failed"
})

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    """Returns the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _load(path: str = LOG_FILE) -> dict:
    """
    Loads run_log.json from disk.

    Returns an empty log structure if the file does not exist.

    Args:
        path: path to run_log.json

    Returns:
        dict with 'version' and 'runs' keys
    """
    if not os.path.exists(path):
        return {"version": "1.0", "runs": []}
    with open(path) as f:
        return json.load(f)


def _save(log: dict, path: str = LOG_FILE) -> None:
    """
    Saves the log dict to disk atomically.

    Args:
        log:  full log dict
        path: destination path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dirpath = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=dirpath, suffix=".tmp", delete=False
    ) as tmp:
        json.dump(log, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _find(log: dict, run_id: str) -> dict:
    """
    Returns the run record for run_id.

    Raises:
        KeyError: if run_id is not in the log
    """
    for run in log["runs"]:
        if run["run_id"] == run_id:
            return run
    raise KeyError(f"run_id not found: {run_id}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register(
    agent_id:      str,
    model:         str,
    age_config:    str,
    scenario_type: str,
    path:          str = LOG_FILE,
) -> str:
    """
    Registers a new conversation run and returns its run_id.

    The run is created with status 'pending'. Call mark_running()
    immediately before making any API calls.

    Args:
        agent_id:      e.g. 'AGENT_A1_EPISTEMIC'
        model:         e.g. 'gpt4o'
        age_config:    one of 'a1' | 'a2' | 'a3'
        scenario_type: one of 'E' | 'S' | 'F'
        path:          path to run_log.json

    Returns:
        str: uuid4 run_id
    """
    run_id = str(uuid.uuid4())
    run = {
        "run_id":            run_id,
        "agent_id":          agent_id,
        "model":             model,
        "age_config":        age_config,
        "scenario_type":     scenario_type,
        "status":            "pending",
        "conversation_path": None,
        "created_at":        _now(),
        "completed_at":      None,
        "scored_at":         None,
        "dvf_computed_at":   None,
        "error":             None,
    }
    log = _load(path)
    log["runs"].append(run)
    _save(log, path)
    return run_id


def mark_running(run_id: str, path: str = LOG_FILE) -> None:
    """
    Sets status to 'running'.

    Args:
        run_id: uuid4 identifier
        path:   path to run_log.json
    """
    log = _load(path)
    _find(log, run_id)["status"] = "running"
    _save(log, path)


def mark_completed(
    run_id:            str,
    conversation_path: str,
    path:              str = LOG_FILE,
) -> None:
    """
    Sets status to 'completed' and records the conversation file path.

    Args:
        run_id:            uuid4 identifier
        conversation_path: path where the conversation JSON was saved
        path:              path to run_log.json
    """
    log = _load(path)
    run = _find(log, run_id)
    run["status"]            = "completed"
    run["conversation_path"] = conversation_path
    run["completed_at"]      = _now()
    _save(log, path)


def mark_scored(run_id: str, path: str = LOG_FILE) -> None:
    """
    Sets status to 'scored' after judge scoring is complete.

    Args:
        run_id: uuid4 identifier
        path:   path to run_log.json
    """
    log = _load(path)
    run = _find(log, run_id)
    run["status"]     = "scored"
    run["scored_at"]  = _now()
    _save(log, path)


def mark_dvf_computed(run_id: str, path: str = LOG_FILE) -> None:
    """
    Sets status to 'dvf_computed' after DVF analysis is complete.

    Args:
        run_id: uuid4 identifier
        path:   path to run_log.json
    """
    log = _load(path)
    run = _find(log, run_id)
    run["status"]          = "dvf_computed"
    run["dvf_computed_at"] = _now()
    _save(log, path)


def mark_failed(
    run_id: str,
    error:  str,
    path:   str = LOG_FILE,
) -> None:
    """
    Sets status to 'failed' and records the error message.

    Args:
        run_id: uuid4 identifier
        error:  error message or traceback string
        path:   path to run_log.json
    """
    log = _load(path)
    run = _find(log, run_id)
    run["status"] = "failed"
    run["error"]  = error
    _save(log, path)


def get_by_status(
    status: str,
    path:   str = LOG_FILE,
) -> list[dict]:
    """
    Returns all runs with the given status.

    Args:
        status: one of pending | running | completed |
                scored | dvf_computed | failed
        path:   path to run_log.json

    Returns:
        list of run dicts

    Raises:
        ValueError: if status is not a recognised value
    """
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Unknown status '{status}'. "
            f"Valid values: {sorted(VALID_STATUSES)}"
        )
    log = _load(path)
    return [r for r in log["runs"] if r["status"] == status]


def get_run(run_id: str, path: str = LOG_FILE) -> dict:
    """
    Returns the run record for run_id.

    Args:
        run_id: uuid4 identifier
        path:   path to run_log.json

    Returns:
        run dict

    Raises:
        KeyError: if run_id is not found
    """
    return _find(_load(path), run_id)


def is_registered(
    agent_id:      str,
    model:         str,
    age_config:    str,
    path:          str = LOG_FILE,
) -> bool:
    """
    Returns True if a non-failed run exists for this (agent, model, age).

    Used by the runner to skip already-completed conversations.

    Args:
        agent_id:   e.g. 'AGENT_A1_EPISTEMIC'
        model:      e.g. 'gpt4o'
        age_config: one of 'a1' | 'a2' | 'a3'
        path:       path to run_log.json

    Returns:
        bool
    """
    log = _load(path)
    for run in log["runs"]:
        if (run["agent_id"]   == agent_id and
                run["model"]      == model and
                run["age_config"] == age_config and
                run["status"]     != "failed"):
            return True
    return False


def summary(path: str = LOG_FILE) -> dict:
    """
    Returns a status count summary of all runs.

    Args:
        path: path to run_log.json

    Returns:
        dict mapping status → count, plus 'total'
    """
    log    = _load(path)
    counts = {s: 0 for s in VALID_STATUSES}
    for run in log["runs"]:
        counts[run["status"]] = counts.get(run["status"], 0) + 1
    counts["total"] = len(log["runs"])
    return counts


def print_summary(path: str = LOG_FILE) -> None:
    """
    Prints a human-readable run summary to stdout.

    Args:
        path: path to run_log.json
    """
    s = summary(path)
    print("=== Run Log Summary ===")
    print(f"  Total:        {s['total']}")
    for status in sorted(VALID_STATUSES):
        if s[status]:
            print(f"  {status:<14}: {s[status]}")
