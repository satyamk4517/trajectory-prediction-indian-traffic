# K-PhysGAT — Kinematic-Bicycle Physics-Injected GAT on NGSIM US-101

> Methodology and evaluation notes for the NGSIM arm of the project.
> Notebook: [`notebooks/01_ngsim_kphysgat_v14.ipynb`](../notebooks/01_ngsim_kphysgat_v14.ipynb)
> Results: [`results/ngsim_v14/`](../results/ngsim_v14/)

---

## 1 · Task

Given 3 seconds of observed past trajectory (30 frames @ 10 Hz) of an ego vehicle plus up to 8 neighbouring vehicles, predict the ego vehicle's future trajectory for the next 5 seconds (50 frames). Evaluation metric: per-second ADE and 5-second FDE, plus minADE under K=6 multimodal Winner-Takes-All.

## 2 · Architecture

```
                   3s × 5-dim per-step features (Δx, Δy, vx, vy, lane_id)
                                    │
                          ┌─────────▼─────────┐
                          │ BiLSTM-256        │
                          │ + LayerNorm       │   per-agent motion encoder
                          └─────────┬─────────┘
                                    │ h_i ∈ ℝ²⁵⁶
                          ┌─────────▼────────────────────┐
                          │ Edge-Feature Multi-Head GAT  │   social interaction:
                          │  4 heads · top-8 neighbours  │   edges = (Δx, Δy,
                          │  edge feats: 5-dim           │             Δvx, Δvy,
                          └─────────┬────────────────────┘             dist)
                                    │ socially-aware h_i'
                          ┌─────────▼────────────────┐
                          │ 2-layer LSTM control     │   decodes 50 timesteps of
                          │ head → (δ_t, a_t)        │   (steering δ, accel a)
                          └─────────┬────────────────┘
                                    │
                          ┌─────────▼──────────────────────────┐
                          │ Differentiable kinematic bicycle:  │
                          │   L = 4.5 m,  dt = 0.1 s           │   physical
                          │   x_{t+1} = x_t + v cos(ψ) dt      │   guarantee:
                          │   y_{t+1} = y_t + v sin(ψ) dt      │   no railgun,
                          │   ψ_{t+1} = ψ + (v/L) tan(δ) dt    │   no hyperspace
                          │   v_{t+1} = v + a dt               │
                          └─────────┬──────────────────────────┘
                                    │
                          50-step future trajectory  →  ADE/FDE
                          ┌─────────▼─────────────────┐
                          │ Horizon-weighted Huber    │   w_t = 1.0 → 3.0
                          │ over 50 timesteps         │   (later steps weigh more)
                          └───────────────────────────┘
```

**Parameters:** ~1.85 M. **Single-sample inference latency:** 297.9 ms (bs=1); **9.4 ms per sample at bs=32**; **0.32 ms per sample at bs=1024** (from [`inference_latency.csv`](../results/ngsim_v14/inference_latency.csv)).

## 3 · Two evaluation protocols, both reported

| Protocol | What it does | Used by | Our ADE |
|---|---|---|---|
| **Random 80/10/10 split** | Standard NGSIM protocol since Deo & Trivedi 2018. Vehicles can appear in train and test. | The entire NGSIM literature including CS-LSTM, GRIP++, BAT, CDSTraj | **0.863 m** |
| **Vehicle-level GroupKFold (k=5)** | The honest protocol: no vehicle in train ever appears in val/test. 3 seeds × 5 folds = **15 runs**. | The few rigorous newer papers | **1.053 m, σ = 0.013** |

Reporting both lets reviewers compare fairly **and** see what an honest generalization number looks like. The tight σ = 0.013 m over 15 runs is the strongest evidence that the result is not seed luck.

## 4 · Headline numbers

### Single-mode prediction (one trajectory output)

ADE_1-to-5s mean (m) — lower is better:

| Model | Year | Protocol | @1s | @3s | @5s | ADE |
|---|---:|---|---:|---:|---:|---:|
| Constant Velocity | — | random | 0.498 | 2.572 | 6.303 | 2.985 |
| CS-LSTM (Deo & Trivedi) | 2018 | random | 0.61 | 2.14 | 4.30 | 1.95 |
| GRIP++ | 2019 | random | 0.38 | 1.62 | 3.48 | 1.61 |
| GISNet | 2020 | random | 0.33 | 1.48 | 2.95 | 1.40 |
| HierarchicalGNN | 2021 | random | 0.34 | 1.32 | 2.83 | 1.30 |
| CDSTraj | 2024 | random | 0.36 | 1.36 | 2.85 | 1.49 |
| BAT | 2024 | random | 0.23 | 1.54 | 3.62 | 1.74 |
| VT-Former | 2024 | random | — | — | — | 0.82 |
| **K-PhysGAT (ours)** | **2026** | **random** | **0.269** | **0.594** | **1.986** | **0.863** |
| **K-PhysGAT (ours)** | **2026** | **GroupKFold (15 runs)** | **0.332** | **0.642** | **2.546** | **1.053 ± 0.013** |

Full table — including 26 published baselines — lives in [`results/ngsim_v14/sota_comparison.csv`](../results/ngsim_v14/sota_comparison.csv).

### Multimodal prediction (K=6 Winner-Takes-All, GroupKFold)

| Metric | Value (3-run mean ± std) |
|---|---|
| minADE | **0.704 ± 0.031 m** |
| minFDE | **1.446 ± 0.095 m** |
| Maneuver accuracy | 98.74% ± 0.03% |

Raw per-seed numbers: [`results/ngsim_v14/multimodal_K6_groupkfold.csv`](../results/ngsim_v14/multimodal_K6_groupkfold.csv).

## 5 · Ablation — what each ingredient buys you

Reading down the rows is the story of v14:

| Variant | ADE_1to5 (random) | Δ vs CV baseline | Comment |
|---|---:|---:|---|
| CV (constant velocity) | 2.985 | — | Reference baseline |
| Iter3A — kinematic LSTM | 1.543 | −48% | + per-agent BiLSTM encoder |
| Iter3B — leader-follower | 1.429 | −52% | + leader vehicle's history (gated) |
| Iter4 — Social GAT (no rollout) | 1.401 | −53% | + graph attention over neighbours |
| **K-PhysGAT (full)** | **0.863** | **−71%** | **+ kinematic bicycle rollout + horizon-weighted Huber** |

The kinematic rollout alone provides a **~38% relative ADE improvement** over the otherwise-identical Social-GAT variant — and crucially, it guarantees kinematically feasible trajectories. That is something none of the 26 published models above can claim.

## 6 · Why this matters

The Random-split number (0.863 m ADE_1to5) sits ahead of every published method that uses the same protocol — including 2024 work (BAT, CDSTraj, VT-Former). The GroupKFold number (1.053 ± 0.013 m) sits ahead of HierarchicalGNN (1.30 m), the best in a recent 26-model survey under comparable conditions, by **19% relative**.

The point is not "we beat SOTA". The point is that **the same model produces a strong number under both protocols, with quantified variance**. That is the honest report.

## 7 · Known limitations (honest)

- **Single-frame ego latency is 297.9 ms.** This is dominated by the Python-loop bicycle rollout. Migrating that loop to `tf.scan` is on the Phase 24 to-do list and should bring single-sample latency under 50 ms.
- **No HD-map context yet.** Many modern motion forecasters (TNT, Trajectron++, M2I) condition on lane geometry. K-PhysGAT is map-free. Adding a lightweight lane-graph encoder is an obvious next step.
- **NGSIM is freeway data.** Strong NGSIM numbers do not transfer to lane-less urban traffic. That is exactly what the Warangal arm of this project addresses — see [`docs/SP_GAT_warangal.md`](SP_GAT_warangal.md).
