# test_dvf.py
# Run this before touching any real data
# python test_dvf.py

import json
from constants import ALPHA, DELTA_ADULT, THETA
import constants

# ----------------------------------------
# Step 1 — Set test constants
# These are placeholder values for testing only
# Replace with calibrated values before real runs
# ----------------------------------------

constants.ALPHA = {
    "a1": {"E": 2.0, "S": 2.5, "F": 1.8},
    "a2": {"E": 1.6, "S": 1.9, "F": 1.5},
    "a3": {"E": 1.2, "S": 1.3, "F": 1.1}
}

constants.DELTA_ADULT = {
    "E": 0.3,
    "S": 0.25,
    "F": 0.2
}

constants.THETA = 0.5

# ----------------------------------------
# Step 2 — Import after setting constants
# ----------------------------------------

from dvf import (
    compute_vt,
    compute_dvf,
    compute_trajectory,
    compute_collapse_point,
    validate_propositions
)

# ----------------------------------------
# Step 3 — Load mock data
# ----------------------------------------

with open("data/sample_scores.json") as f:
    data = json.load(f)

# ----------------------------------------
# Step 4 — Validate propositions
# ----------------------------------------

print("=== Validating Theoretical Propositions ===")
results = validate_propositions()
for prop, satisfied in results.items():
    status = "PASS" if satisfied else "FAIL"
    print(f"  {prop}: {status}")

# ----------------------------------------
# Step 5 — Compute DVF for each conversation
# ----------------------------------------

print("\n=== DVF Computation Test ===")
for conv in data["conversations"]:
    model = conv["model"]
    age = conv["age_config"]
    turn_scores = conv["turn_scores"]
    
    dvf_score = compute_dvf(turn_scores, age)
    trajectory = compute_trajectory(turn_scores, age)
    collapse = compute_collapse_point(turn_scores, age)
    
    print(f"\nModel: {model} | Age: {age}")
    print(f"  DVF Score: {dvf_score:.4f}")
    print(f"  Trajectory: "
          f"{[round(v,3) for v in trajectory]}")
    print(f"  Collapse Point: "
          f"Turn {collapse if collapse is not None else 'None'}")

# ----------------------------------------
# Step 6 — Validate P1 ordering on mock data
# ----------------------------------------

print("\n=== Prediction P1 Validation (Mock Data) ===")
scores = {}
for conv in data["conversations"]:
    age = conv["age_config"]
    score = compute_dvf(conv["turn_scores"], age)
    scores[age] = score

p1_confirmed = scores["a1"] > scores["a2"] > scores["a3"]
print(f"  a1: {scores['a1']:.4f}")
print(f"  a2: {scores['a2']:.4f}")
print(f"  a3: {scores['a3']:.4f}")
print(f"  P1 Confirmed: {p1_confirmed}")

print("\n=== All tests complete ===")