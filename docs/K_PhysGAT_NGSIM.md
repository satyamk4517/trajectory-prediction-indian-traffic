# K-PhysGAT — Kinematic-Bicycle Physics-Injected GAT on NGSIM US-101

> Methodology and evaluation notes for the NGSIM arm of the project.
> Notebook: [`notebooks/01_ngsim_kphysgat_v14.ipynb`](../notebooks/01_ngsim_kphysgat_v14.ipynb)
> Results: [`results/ngsim_v14/`](../results/ngsim_v14/)
>
> **Role in the project.** K-PhysGAT is the NGSIM-side control model — the earlier phase of
> this project, kept alive specifically as a **protocol-control study**, not as a benchmark
> entry. The current, paper-verified model for the project's own Indian-traffic dataset is
> **HetSpA** (Warangal); see [`docs/SP_GAT_warangal.md`](SP_GAT_warangal.md). A manuscript
> covering the HetSpA work is in preparation for *Transportation Research Part C*.
>
> **Non-comparability caveat (applies to every number on this page).** This NGSIM pipeline
> uses pre-smoothed 10 Hz data and a different split from the published literature, so these
> numbers are **not line-comparable to published results**, and no ranking against them is
> claimed anywhere in this document.

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

## 3 · The split-inflation finding (the actual story of this page)

K-PhysGAT is trained and evaluated once, then scored under **two different train/test splits**
of the same data. The gap between the two scores is the point of this study.

| Protocol | What it does | Used by | K-PhysGAT ADE_1to5 |
|---|---|---|---:|
| **Random 80/10/10 split** | The field's customary NGSIM protocol since Deo & Trivedi 2018. Vehicles can appear in both train and test. | Most of the published NGSIM literature | **0.863 m** |
| **Vehicle-level GroupKFold (k=5)** | The honest protocol: no vehicle in train ever appears in val/test. 3 seeds × 5 folds = **15 runs**. | A minority of newer, more careful papers | **1.053 ± 0.013 m** |

The **same model, same weights, same test set of raw trajectories** scores 0.863 m under the
customary split and 1.053 ± 0.013 m once vehicle identity can no longer leak from train into
test — an **18% inflation** attributable to vehicle-identity leakage alone. The 15-run
standard deviation (σ = 0.013 m) shows this gap is not seed noise; it is a property of the
split.

**Reading this finding correctly:** it is not a claim that K-PhysGAT is unusually leaky — it is
a claim that *the customary protocol itself* flatters any model trained and evaluated this way,
because a vehicle's own trajectory in the training set is informative about that same vehicle's
trajectory in the test set. Published numbers that use the customary random split rest on this
same inflation; this repository cannot say by how much, because that would require rerunning
those models under GroupKFold, which is out of scope here. The controlled point is narrower and
fully within scope: the same architecture, the same data, two protocols, one honest delta.

## 4 · Headline numbers (K-PhysGAT's own results only)

### Single-mode prediction (one trajectory output)

ADE per second (m) — lower is better. Both rows are the same model, same weights.

| Protocol | @1s | @2s | @3s | @4s | @5s | ADE_1to5 |
|---|---:|---:|---:|---:|---:|---:|
| Random split | 0.269 | 0.336 | 0.594 | 1.129 | 1.986 | **0.863** |
| GroupKFold (15-run mean ± sd) | 0.332 | 0.399 | 0.642 | 1.344 | 2.546 | **1.053 ± 0.013** |

Source: [`results/ngsim_v14/all_results_random_and_groupkfold.json`](../results/ngsim_v14/all_results_random_and_groupkfold.json) and [`results/ngsim_v14/summary_15run_variance.json`](../results/ngsim_v14/summary_15run_variance.json).

### Multimodal prediction (K=6 Winner-Takes-All, GroupKFold)

| Metric | Value (3-seed mean ± std) |
|---|---|
| minADE | **0.704 ± 0.031 m** |
| minFDE | **1.446 ± 0.095 m** |
| Maneuver accuracy | 98.74% ± 0.03% |

Raw per-seed numbers: [`results/ngsim_v14/multimodal_K6_groupkfold.csv`](../results/ngsim_v14/multimodal_K6_groupkfold.csv).

*Caveat repeated: these are GroupKFold numbers on pre-smoothed 10 Hz NGSIM data under this
project's own split — not comparable to published multimodal NGSIM results.*

## 5 · Internal ablation — what each ingredient buys you (random split, our own variants only)

This is a lesion study against the project's own naive baselines, not a comparison to
published models.

| Variant | ADE_1to5 (random) | Δ vs CV baseline | Comment |
|---|---:|---:|---|
| CV (constant velocity) | 2.985 | — | Reference baseline |
| Iter3A — kinematic LSTM | 1.543 | −48% | + per-agent BiLSTM encoder |
| Iter3B — leader-follower | 1.429 | −52% | + leader vehicle's history (gated) |
| Iter4 — Social GAT (no rollout) | 1.401 | −53% | + graph attention over neighbours |
| **K-PhysGAT (full)** | **0.863** | **−71%** | **+ kinematic bicycle rollout + horizon-weighted Huber** |

The kinematic rollout alone accounts for a **~38% relative ADE improvement** over the
otherwise-identical Social-GAT variant (Iter4), and it guarantees every output trajectory is
kinematically feasible by construction — the rollout integrates bounded steering/acceleration
controls rather than regressing raw (x, y) points.

## 6 · Why this page exists

The point of this page is **not** a ranking against published NGSIM methods. The point is the
split-inflation measurement in §3: the same model, same data, two protocols, one 18% honest
gap. That gap is the evidence this repository can actually stand behind, and it is the reason
the Warangal arm of the project (see [`docs/SP_GAT_warangal.md`](SP_GAT_warangal.md)) evaluates
HetSpA under vehicle-level GroupKFold from the start rather than retrofitting it later.

No sentence on this page ranks K-PhysGAT against a published NGSIM number. Any such comparison
would need matched preprocessing (this pipeline uses pre-smoothed 10 Hz data), a matched split,
and a matched horizon before it could be asserted — none of which is verified here.

## 7 · Known limitations (honest)

- **Single-frame ego latency is 297.9 ms.** This is dominated by the Python-loop bicycle rollout. Migrating that loop to `tf.scan` is on the future-work list and should bring single-sample latency well under 100 ms.
- **No HD-map context.** Many modern motion forecasters condition on lane geometry. K-PhysGAT is map-free.
- **NGSIM is freeway data.** Strong or weak NGSIM numbers do not tell you anything about lane-free urban traffic. That is exactly what the Warangal arm of this project addresses — see [`docs/SP_GAT_warangal.md`](SP_GAT_warangal.md), the current HetSpA documentation.
- **This page's numbers are a protocol-control study, not a leaderboard entry.** Treat §3 as the finding; treat §4–§5 as the supporting detail.
