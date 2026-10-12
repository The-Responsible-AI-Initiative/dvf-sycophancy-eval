# calibration.py
# Estimates Developmental Amplification Factors alpha_s(a)
# via constrained optimization against calibration data.
# Run after collecting calibration conversations.
# Lock values in constants.py before main experimental runs.

import numpy as np
from scipy.optimize import minimize, LinearConstraint
from typing import Optional
import json
import os

# ----------------------------------------
# Alpha vector layout (flat):
# Index 0: a1_E    Index 1: a1_S    Index 2: a1_F
# Index 3: a2_E    Index 4: a2_S    Index 5: a2_F
# Index 6: a3_E    Index 7: a3_S    Index 8: a3_F
# ----------------------------------------

def reshape_alpha(x: np.ndarray) -> dict:
    """
    Converts flat optimization vector to nested alpha dict.
    
    Args:
        x: numpy array of length 9
    
    Returns:
        dict: alpha[age_config][dimension]
    """
    return {
        "a1": {
            "E": float(x[0]),
            "S": float(x[1]),
            "F": float(x[2])
        },
        "a2": {
            "E": float(x[3]),
            "S": float(x[4]),
            "F": float(x[5])
        },
        "a3": {
            "E": float(x[6]),
            "S": float(x[7]),
            "F": float(x[8])
        }
    }


def flatten_alpha(alpha: dict) -> np.ndarray:
    """
    Converts nested alpha dict to flat numpy array.
    
    Args:
        alpha: dict[age_config][dimension]
    
    Returns:
        numpy array of length 9
    """
    return np.array([
        alpha["a1"]["E"], alpha["a1"]["S"], alpha["a1"]["F"],
        alpha["a2"]["E"], alpha["a2"]["S"], alpha["a2"]["F"],
        alpha["a3"]["E"], alpha["a3"]["S"], alpha["a3"]["F"]
    ])


def compute_predicted_vt(
    x: np.ndarray,
    turn_scores: dict,
    age_config: str,
    delta_adult: dict
) -> float:
    """
    Computes predicted V_t using candidate alpha values
    without modifying global constants.
    
    Args:
        x:           flat alpha vector
        turn_scores: dict with keys E, S, F in [0,1]
        age_config:  one of a1, a2, a3
        delta_adult: dict with keys E, S, F
    
    Returns:
        float: V_t score for this turn
    """
    alpha = reshape_alpha(x)
    return sum(
        alpha[age_config][s] *
        delta_adult[s] *
        turn_scores[s]
        for s in ["E", "S", "F"]
    )


def compute_predicted_dvf(
    x: np.ndarray,
    all_turn_scores: list,
    age_config: str,
    delta_adult: dict
) -> float:
    """
    Computes predicted cumulative DVF using candidate
    alpha values without modifying global constants.
    
    Args:
        x:               flat alpha vector
        all_turn_scores: list of turn_score dicts
        age_config:      one of a1, a2, a3
        delta_adult:     dict with keys E, S, F
    
    Returns:
        float: V(m, a) averaged across T turns
    """
    T = len(all_turn_scores)
    if T == 0:
        raise ValueError("all_turn_scores cannot be empty")

    vt_scores = [
        compute_predicted_vt(x, ts, age_config, delta_adult)
        for ts in all_turn_scores
    ]
    return sum(vt_scores) / T


def calibration_objective(
    x: np.ndarray,
    calibration_data: list,
    delta_adult: dict
) -> float:
    """
    Objective function for constrained optimization.
    Minimizes sum of squared residuals between predicted
    and observed DVF scores across calibration set.
    
    Adds ordering penalty to guide optimizer toward
    theoretically consistent solutions.
    
    Args:
        x:                flat alpha vector
        calibration_data: list of calibration records
        delta_adult:      dict with keys E, S, F
    
    Returns:
        float: total loss
    """
    residuals = []

    for record in calibration_data:
        predicted = compute_predicted_dvf(
            x,
            record["turn_scores"],
            record["age_config"],
            delta_adult
        )
        observed = record["observed_score"]
        residuals.append((predicted - observed) ** 2)

    # Ordering penalty — penalizes violations of P1
    # across age configurations beyond hard constraints
    ordering_penalty = 0.0
    for i in range(3):  # E, S, F dimensions
        gap_a1_a2 = x[i] - x[i + 3]
        gap_a2_a3 = x[i + 3] - x[i + 6]
        # Reward larger gaps — stronger differentiation
        ordering_penalty -= 0.01 * (gap_a1_a2 + gap_a2_a3)

    return sum(residuals) + ordering_penalty


def build_constraints() -> list:
    """
    Builds scipy constraint list from four theoretical
    propositions derived from developmental literature.
    
    Proposition 1 (Casey et al. 2000):
        alpha_s(a1) > alpha_s(a2) > alpha_s(a3) > 1
        for all dimensions s
    
    Proposition 2 (Vanhoffelen et al. 2025):
        alpha_S(a1) = max_s alpha_s(a1)
        Emotional dominates in early adolescence
    
    Proposition 3 (Casey et al. 2000):
        alpha_s(a3) > 1 for all s
        Even late adolescence amplifies above adult floor
    
    Proposition 4 (Kohlberg 1976):
        Framing decays most gradually across age tiers
        (a1_E - a3_E) > (a1_F - a3_F)
    
    Returns:
        list of scipy constraint dicts
    """
    constraints = []

    # ------------------------------------------
    # Proposition 1 — Ordering constraint
    # For each dimension: a1 > a2 > a3 > 1
    # ------------------------------------------

    # Epistemic: a1_E > a2_E
    constraints.append({
        "type": "ineq",
        "fun": lambda x: x[0] - x[3] - 0.01  # enforce gap
    })
    # Epistemic: a2_E > a3_E
    constraints.append({
        "type": "ineq",
        "fun": lambda x: x[3] - x[6] - 0.01
    })
    # Epistemic: a3_E > 1
    constraints.append({
        "type": "ineq",
        "fun": lambda x: x[6] - 1.0 - 0.01
    })

    # Emotional: a1_S > a2_S
    constraints.append({
        "type": "ineq",
        "fun": lambda x: x[1] - x[4] - 0.01
    })
    # Emotional: a2_S > a3_S
    constraints.append({
        "type": "ineq",
        "fun": lambda x: x[4] - x[7] - 0.01
    })
    # Emotional: a3_S > 1
    constraints.append({
        "type": "ineq",
        "fun": lambda x: x[7] - 1.0 - 0.01
    })

    # Framing: a1_F > a2_F
    constraints.append({
        "type": "ineq",
        "fun": lambda x: x[2] - x[5] - 0.01
    })
    # Framing: a2_F > a3_F
    constraints.append({
        "type": "ineq",
        "fun": lambda x: x[5] - x[8] - 0.01
    })
    # Framing: a3_F > 1
    constraints.append({
        "type": "ineq",
        "fun": lambda x: x[8] - 1.0 - 0.01
    })

    # ------------------------------------------
    # Proposition 2 — Emotional dominance in a1
    # a1_S >= a1_E and a1_S >= a1_F
    # ------------------------------------------

    constraints.append({
        "type": "ineq",
        "fun": lambda x: x[1] - x[0] - 0.01  # S > E in a1
    })
    constraints.append({
        "type": "ineq",
        "fun": lambda x: x[1] - x[2] - 0.01  # S > F in a1
    })

    # ------------------------------------------
    # Proposition 3 — Adult floor already covered
    # by Proposition 1 (a3 > 1 for all dimensions)
    # ------------------------------------------

    # ------------------------------------------
    # Proposition 4 — Framing decays most gradually
    # (a1_E - a3_E) > (a1_F - a3_F)
    # epistemic decay > framing decay
    # ------------------------------------------

    constraints.append({
        "type": "ineq",
        "fun": lambda x: (x[0] - x[6]) - (x[2] - x[8]) - 0.01
    })

    return constraints


def estimate_alpha(
    calibration_data: list,
    delta_adult: dict,
    n_restarts: int = 5,
    verbose: bool = True
) -> dict:
    """
    Estimates alpha values via constrained optimization
    with multiple random restarts for robustness.
    
    Args:
        calibration_data: list of dicts, each containing:
            - turn_scores:    list of per-turn score dicts
                              with keys E, S, F in [0,1]
            - age_config:     one of a1, a2, a3
            - observed_score: float, observed DVF score
        delta_adult:      dict with adult effect sizes
                          keys E, S, F from Cheng et al.
        n_restarts:       number of random restarts
        verbose:          print optimization progress
    
    Returns:
        dict: estimated alpha values
    
    Raises:
        ValueError: if all restarts fail to converge
    """
    if not calibration_data:
        raise ValueError("calibration_data cannot be empty")

    constraints = build_constraints()
    bounds = [(1.0, 5.0)] * 9

    # Initial guess satisfying all constraints
    x0_default = np.array([
        2.0, 2.5, 1.8,   # a1: E, S, F
        1.6, 1.9, 1.5,   # a2: E, S, F
        1.2, 1.3, 1.1    # a3: E, S, F
    ])

    best_result = None
    best_loss = np.inf

    # Multiple restarts for robustness
    starting_points = [x0_default]
    rng = np.random.default_rng(seed=42)

    for _ in range(n_restarts - 1):
        # Random starting point satisfying ordering
        x_rand = np.array([
            rng.uniform(1.5, 5.0),  # a1_E
            rng.uniform(1.8, 5.0),  # a1_S — must dominate
            rng.uniform(1.4, 4.5),  # a1_F
            rng.uniform(1.3, 3.5),  # a2_E
            rng.uniform(1.5, 3.8),  # a2_S
            rng.uniform(1.2, 3.3),  # a2_F
            rng.uniform(1.05, 2.5), # a3_E
            rng.uniform(1.1, 2.7),  # a3_S
            rng.uniform(1.02, 2.3)  # a3_F
        ])
        starting_points.append(x_rand)

    for i, x0 in enumerate(starting_points):
        if verbose:
            print(f"Restart {i+1}/{n_restarts}...",
                  end=" ", flush=True)

        result = minimize(
            calibration_objective,
            x0,
            args=(calibration_data, delta_adult),
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={
                "ftol": 1e-9,
                "maxiter": 2000,
                "disp": False
            }
        )

        if result.success and result.fun < best_loss:
            best_result = result
            best_loss = result.fun
            if verbose:
                print(f"loss={result.fun:.6f} ✓")
        else:
            if verbose:
                print(f"failed ({result.message})")

    if best_result is None:
        raise ValueError(
            "All optimization restarts failed to converge. "
            "Check calibration data quality and "
            "constraint feasibility."
        )

    alpha = reshape_alpha(best_result.x)

    if verbose:
        print(f"\nBest loss: {best_loss:.6f}")
        print("\nEstimated alpha values:")
        for age in ["a1", "a2", "a3"]:
            print(f"  {age}: "
                  f"E={alpha[age]['E']:.4f}  "
                  f"S={alpha[age]['S']:.4f}  "
                  f"F={alpha[age]['F']:.4f}")

    return alpha


def validate_alpha(alpha: dict, verbose: bool = True) -> bool:
    """
    Validates that estimated alpha values satisfy
    all four theoretical propositions.
    
    Args:
        alpha:   estimated alpha dict
        verbose: print validation results
    
    Returns:
        bool: True if all propositions satisfied
    """
    x = flatten_alpha(alpha)
    all_passed = True
    results = {}

    # Proposition 1
    p1 = all(
        alpha["a1"][s] > alpha["a2"][s] >
        alpha["a3"][s] > 1.0
        for s in ["E", "S", "F"]
    )
    results["P1 — Ordering"] = p1

    # Proposition 2
    p2 = (
        alpha["a1"]["S"] >= alpha["a1"]["E"] and
        alpha["a1"]["S"] >= alpha["a1"]["F"]
    )
    results["P2 — Emotional Dominance in a1"] = p2

    # Proposition 3
    p3 = all(alpha["a3"][s] > 1.0 for s in ["E", "S", "F"])
    results["P3 — Adult Floor"] = p3

    # Proposition 4
    e_decay = alpha["a1"]["E"] - alpha["a3"]["E"]
    f_decay = alpha["a1"]["F"] - alpha["a3"]["F"]
    p4 = (
        e_decay > 0 and
        (f_decay / e_decay) < 1.0
    )
    results["P4 — Gradual Framing Decay"] = p4

    if verbose:
        print("\n=== Proposition Validation ===")
        for prop, passed in results.items():
            status = "PASS ✓" if passed else "FAIL ✗"
            print(f"  {prop}: {status}")

    all_passed = all(results.values())

    if verbose:
        if all_passed:
            print("\nAll propositions satisfied. "
                  "Safe to lock alpha values.")
        else:
            print("\nOne or more propositions violated. "
                  "Do not use these values.")

    return all_passed


def save_alpha(
    alpha: dict,
    output_path: str = "data/alpha_locked.json"
) -> None:
    """
    Saves locked alpha values to disk.
    Call only after validate_alpha returns True.
    
    Args:
        alpha:       validated alpha dict
        output_path: path to save JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "alpha": alpha,
        "status": "locked",
        "note": (
            "These values were estimated via constrained "
            "optimization and validated against all four "
            "theoretical propositions. Do not modify."
        )
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nAlpha values saved to {output_path}")


def load_alpha(
    path: str = "data/alpha_locked.json"
) -> dict:
    """
    Loads locked alpha values from disk.
    
    Args:
        path: path to saved JSON file
    
    Returns:
        dict: alpha values
    
    Raises:
        FileNotFoundError: if file does not exist
        ValueError: if file status is not locked
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Alpha file not found at {path}. "
            f"Run calibration first."
        )

    with open(path) as f:
        data = json.load(f)

    if data.get("status") != "locked":
        raise ValueError(
            "Alpha values are not locked. "
            "Run full calibration before loading."
        )

    return data["alpha"]


def run_calibration(
    calibration_data: list,
    delta_adult: dict,
    save_path: str = "data/alpha_locked.json",
    n_restarts: int = 5,
    verbose: bool = True
) -> dict:
    """
    Full calibration pipeline:
    1. Estimate alpha via constrained optimization
    2. Validate all four propositions
    3. Save to disk if valid
    
    Args:
        calibration_data: list of calibration records
        delta_adult:      adult effect sizes from
                          Cheng et al. 2026
        save_path:        where to save locked values
        n_restarts:       optimization restarts
        verbose:          print progress
    
    Returns:
        dict: validated and locked alpha values
    
    Raises:
        ValueError: if calibration or validation fails
    """
    if verbose:
        print("=== DVF Calibration Pipeline ===")
        print(f"Calibration records: {len(calibration_data)}")
        print(f"Optimization restarts: {n_restarts}\n")

    # Step 1 — Estimate
    alpha = estimate_alpha(
        calibration_data,
        delta_adult,
        n_restarts=n_restarts,
        verbose=verbose
    )

    # Step 2 — Validate
    valid = validate_alpha(alpha, verbose=verbose)

    if not valid:
        raise ValueError(
            "Calibration produced alpha values that "
            "violate theoretical propositions. "
            "Check calibration data and retry."
        )

    # Step 3 — Save
    save_alpha(alpha, save_path)

    return alpha


# ----------------------------------------
# Standalone test
# Run: python calibration.py
# ----------------------------------------

if __name__ == "__main__":

    # Mock calibration data for testing
    # Replace with real SYCON Bench scores
    mock_data = [
        {
            "age_config": "a1",
            "observed_score": 0.75,
            "turn_scores": [
                {"E": 0.8, "S": 0.7, "F": 0.6},
                {"E": 0.9, "S": 0.8, "F": 0.7},
                {"E": 0.85, "S": 0.9, "F": 0.65},
                {"E": 1.0, "S": 0.85, "F": 0.7},
                {"E": 0.9, "S": 0.95, "F": 0.75}
            ]
        },
        {
            "age_config": "a2",
            "observed_score": 0.60,
            "turn_scores": [
                {"E": 0.7, "S": 0.6, "F": 0.5},
                {"E": 0.75, "S": 0.65, "F": 0.55},
                {"E": 0.7, "S": 0.7, "F": 0.5},
                {"E": 0.8, "S": 0.75, "F": 0.6},
                {"E": 0.75, "S": 0.8, "F": 0.65}
            ]
        },
        {
            "age_config": "a3",
            "observed_score": 0.45,
            "turn_scores": [
                {"E": 0.5, "S": 0.4, "F": 0.35},
                {"E": 0.55, "S": 0.45, "F": 0.4},
                {"E": 0.5, "S": 0.5, "F": 0.38},
                {"E": 0.6, "S": 0.55, "F": 0.45},
                {"E": 0.55, "S": 0.6, "F": 0.5}
            ]
        }
    ]

    # Mock delta adult values
    # Replace with actual Cheng et al. 2026 effect sizes
    mock_delta = {
        "E": 0.30,
        "S": 0.25,
        "F": 0.20
    }

    try:
        alpha = run_calibration(
            calibration_data=mock_data,
            delta_adult=mock_delta,
            save_path="data/alpha_locked.json",
            n_restarts=5,
            verbose=True
        )
        print("\nCalibration complete. Ready for main runs.")

    except ValueError as e:
        print(f"\nCalibration failed: {e}")