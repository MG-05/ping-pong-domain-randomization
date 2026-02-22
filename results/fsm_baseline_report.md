# FSM/IK Baseline Controller — Evaluation Report

**Date:** 2026-02-21
**Controller:** `src/controllers/fsm_controller.py` (FSMController)
**Simulation:** Drake, nominal physics (no domain randomization)
**Evaluation script:** `src/evaluate.py` (or via `PingPongResidualEnv` with zero action)
**Raw data:** `results/fsm_baseline_eval.json`

---

## Overview

The FSM/IK controller is a finite-state-machine that predicts ball intercept
times, solves inverse kinematics for paddle positioning, and executes
timed strike–follow-through motions. It requires no learned components.

This report evaluates the controller across three conditions:

1. **Deterministic** — fixed ball initial position (0.77, -0.03, 0.70) m
2. **Noisy** — ball start randomised ±0.04 m per axis (50 episodes)
3. **Stress** — ball start randomised ±0.08 m per axis (50 episodes)

All episodes run for up to 20 s of simulation (2 000 control steps at 10 ms).

---

## 1. Deterministic Performance

| Metric | Value |
|---|---|
| Total hits | 154 |
| Simulation time | 19.3 s |
| Hit rate | 8.0 hits/s |
| Mean hit interval | 120 ms (σ = 168 ms) |
| Apexes detected | 46 |
| Mean apex height | 0.557 m (target 0.55) |
| Apex std | 0.030 m |
| Apex MAE | 0.019 m |
| Apex range | [0.528, 0.663] m |

Under ideal initial conditions the controller sustains a near-continuous
rally for the full simulation. Apex tracking is accurate on average but has
occasional outliers up to +11 cm above target.

---

## 2. Noisy Initial Conditions (±0.04 m)

50 episodes, seeds 0–49.

### Rally statistics

| Metric | Value |
|---|---|
| Mean hits | 112.3 ± 46.7 |
| Median hits | 143 |
| Min / Max hits | 33 / 168 |
| Mean rally duration | 14.3 ± 6.6 s |
| Full 20 s survival | **25 / 50 (50 %)** |

### Apex tracking (1 711 apex samples)

| Metric | Value |
|---|---|
| Mean apex height | 0.552 ± 0.031 m |
| Apex MAE | 0.018 m |
| Within ±2 cm of target | 75.6 % |
| Within ±5 cm of target | 93.2 % |
| Apex range | [0.357, 0.719] m |

### Failure analysis

| | Count | Mean hits | Mean duration |
|---|---|---|---|
| **Failed** (< 20 s) | 25 / 50 | 72.6 ± 34.0 | 8.5 s |
| — failed < 5 hits | 0 | — | — |
| — failed 5–50 hits | 9 | — | — |
| — failed 50+ hits | 16 | — | — |
| **Survived** (≥ 20 s) | 25 / 50 | 152.1 ± 6.7 | 20.0 s |

The controller behaviour is **bimodal**: episodes either lock into a stable
rally (≈152 hits) or diverge and lose the ball partway through (≈73 hits).
All episodes achieve at least 33 hits before any failure.

---

## 3. Stress Test (±0.08 m, 2× nominal noise)

50 episodes, seeds 0–49.

| Metric | Value |
|---|---|
| Mean hits | 104.3 ± 54.9 |
| Min / Max hits | 25 / 163 |
| Full 20 s survival | **23 / 50 (46 %)** |
| Mean rally duration | 13.1 ± 7.3 s |

Doubling the initial-condition noise barely degrades the survival rate
(50 % → 46 %), suggesting failures depend more on specific ball-trajectory
geometries than on noise magnitude.

---

## Key Findings

1. **Strong deterministic baseline.** The FSM/IK controller reliably
   sustains 154 hits at 8 hits/s under nominal conditions with good
   apex accuracy (MAE = 19 mm).

2. **50 % survival under noise.** With realistic ±4 cm initial-condition
   variation, exactly half the episodes survive the full 20 s rally.

3. **Bimodal behaviour.** Success and failure are sharply separated:
   surviving episodes average 152 hits; failing episodes average 73 hits.
   There is no gradual degradation.

4. **Apex tracking is consistent when rallies survive.** 93 % of observed
   apexes fall within ±5 cm of the 0.55 m target. The worst outlier
   is 0.36 m (19 cm below target).

5. **Robust to noise magnitude.** Doubling the noise from ±4 cm to ±8 cm
   only reduces survival from 50 % to 46 %, indicating failures are
   geometry-dependent, not noise-scale-dependent.

---

## Reproducing These Results

```bash
# Evaluate FSM baseline (zero residual) for 50 episodes
python -m src.evaluate --fsm-only --episodes 50

# Or run the detailed evaluation script used for this report:
# (see the Python snippet that generated results/fsm_baseline_eval.json)
```
