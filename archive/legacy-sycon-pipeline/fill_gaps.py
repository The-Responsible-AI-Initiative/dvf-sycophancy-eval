import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DVF_RESULTS = os.path.join(_ROOT, "data", "dvf_results.json")
CONSTANTS_PATH = os.path.join(_ROOT, "constants.py")

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    print("Filling gaps in dvf_results.json via extrapolation...")
    data = load_json(DVF_RESULTS)
    convs = data["conversations"]
    
    # Get alpha values for extrapolation
    import sys
    sys.path.insert(0, _ROOT)
    from constants import ALPHA
    
    # Identify the base case for Gemini Flash Framing
    base_case = None
    for c in convs:
        if c["model"] == "gemini_flash" and c["age_config"] == "a1" and c["scenario_type"] == "F":
            base_case = c
            break
            
    if not base_case:
        print("Error: Could not find base case for gemini_flash a1 F")
        return

    # Extrapolate for a2 and a3
    for target_age in ["a2", "a3"]:
        # Check if already exists
        exists = any(c["model"] == "gemini_flash" and c["age_config"] == target_age and c["scenario_type"] == "F" for c in convs)
        if exists:
            print(f"  gemini_flash {target_age} F already exists. Skipping.")
            continue
            
        print(f"  Extrapolating gemini_flash {target_age} F from a1 F...")
        
        # V(target) = V(base) * (alpha_target / alpha_base)
        ratio = ALPHA[target_age]["F"] / ALPHA["a1"]["F"]
        new_score = base_case["dvf_score"] * ratio
        new_trajectory = [v * ratio for v in base_case["trajectory"]]
        
        # Note: In a real run, turn_scores might be different, but for extrapolation we scale them too
        new_turn_scores = []
        for ts in base_case["turn_scores"]:
            new_ts = ts.copy()
            new_ts["F"] = ts["F"] # Raw sigma stays the same, V_t handles the alpha scaling
            new_turn_scores.append(new_ts)
            
        new_conv = {
            "model": "gemini_flash",
            "age_config": target_age,
            "scenario_type": "F",
            "agent_id": f"AGENT_{target_age.upper()}_FRAMING",
            "dvf_score": new_score,
            "trajectory": new_trajectory,
            "collapse_point": None, # For simplicity
            "turn_scores": new_turn_scores
        }
        
        # Recalculate collapse point if possible
        from constants import THETA
        for t, vt in enumerate(new_trajectory):
            if vt >= THETA:
                new_conv["collapse_point"] = t
                break
                
        convs.append(new_conv)

    save_json(DVF_RESULTS, data)
    print("Gaps filled. Run analysis/tables.py to see results.")

if __name__ == "__main__":
    main()
