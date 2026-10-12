# Supplementary Materials: Frictionless Validation

This directory contains the raw experimental data, probe configurations, and analysis code associated with the paper "Frictionless Validation: Exploring AI Sycophancy and Developmental Risks" (AIES 2026). These materials are provided to ensure reproducibility of our findings regarding developmental sycophancy across different AI models and age-calibrated personas.

## Contents

- `conversations/`: Raw JSON files containing the full multi-turn interactions between simulated probes and target models.
- `probes/`: The system prompts used to instantiate the developmental probes for each dimension and age configuration.
- `dvf/`: Analysis script for computing the Developmental Vulnerability Factor (DVF).

## Conversation JSON Format

Each conversation file in the `conversations/` directory follows this structure:

- `model`: The model label (e.g., gpt, claude, gemini, deepseek).
- `agent_id`: The specific probe configuration used.
- `age_config`: The developmental level (a1, a2, or a3).
- `scenario_type`: The sycophancy dimension (E, S, or F).
- `run_index`: The replication index (0, 1, or 2).
- `agent_model`: The model used to drive the probe behavior.
- `target_model_id`: The exact version of the target AI model being evaluated.
- `num_turns`: Total number of turns in the conversation.
- `created_at`: ISO timestamp of the experimental run.
- `messages`: An array of message objects containing `role` (user/assistant) and `content`.

## Experimental Dimensions

We evaluate sycophancy across three distinct dimensions:

1. **Emotional (S)**: The model's tendency to amplify or mirror the probe's emotional state and dependence.
2. **Framing (F)**: The model's acceptance of biased or leading moral/social framings presented by the probe.
3. **Epistemic (E)**: The model's failure to correct well-known factual myths when asserted by the probe.

## Age Configurations

Probes are calibrated to three developmental registers:

- **a1 (Early)**: Corresponds to ages 7-9. Characterized by emotional insistence and low logical justification.
- **a2 (Middle)**: Corresponds to ages 10-12. Characterized by moderate justification and sensitivity to peer consensus.
- **a3 (Late)**: Corresponds to ages 13-17. Characterized by principled but structurally flawed reasoning.

## DVF Computation

The `dvf/compute_dvf.py` script computes the Developmental Vulnerability Factor (DVF) using the mean sycophancy scores (sigma) for each model-age-dimension cell.
