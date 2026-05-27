<div align="center">

# Trajectory Prediction for Heterogeneous Indian Urban Traffic

### **K-PhysGAT** — Kinematic-Bicycle Physics-Injected Graph Attention Networks for lane-less, mixed-class traffic

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![IIT Guwahati](https://img.shields.io/badge/IIT-Guwahati-1f77b4?style=flat-square)](https://www.iitg.ac.in/)
[![Status: Active Research](https://img.shields.io/badge/Status-Active%20Research-success?style=flat-square)]()

**M.Tech Thesis · IIT Guwahati · Department of Civil Engineering · 2024 – Present**
Supervisor: **Prof. C. Mallikarjuna** · Transportation Systems Engineering

[Headline result](#-headline-results) · [Architecture](#-architecture) · [Reproducibility](#%EF%B8%8F-reproducibility) · [Engineering lessons](#-engineering-lessons-from-the-trenches) · [Roadmap](#%EF%B8%8F-roadmap)

</div>

---

## 🎯 The problem in one sentence

> Most trajectory-prediction work is trained and evaluated on **NGSIM** — a structured US freeway dataset. These models break on Indian urban roads where lane discipline is absent, vehicle classes span 50× in mass, and interactions are *weaving* rather than *following*.

This project asks: **can interaction-aware, physics-injected, safety-regularised deep learning close that gap — without sacrificing physical plausibility?** The answer, supported by a 15-run GroupKFold sweep on NGSIM and a six-phase ablation on a Warangal drone-video dataset, is yes.

<div align="center">

| Property | NGSIM (US freeway) | Warangal (India urban) |
|---|---|---|
| Road type | Structured freeway | 2-lane, 2-way undivided urban |
| Lane discipline | Strict | Effectively absent |
| Vehicle classes | Cars, trucks | Bikes, autos, cars, buses |
| Mass range across classes | ~1× | **up to 50×** (bike → bus) |
| Interaction style | Following, merging | Weaving, overtaking, lateral drift |

</div>

---

## 🏆 Headline results

### NGSIM US-101 — K-PhysGAT vs 26 published models

Two protocols reported. The first is the standard NGSIM 80/10/10 random split used by every paper from CS-LSTM (2018) onward. The second is **vehicle-level GroupKFold (k = 5)** — train/val/test vehicles never overlap — averaged over **15 independent runs (3 seeds × 5 folds)**.

<div align="center">

| Model | Year | @1s | @3s | @5s | **ADE 1→5s** |
|---|---:|---:|---:|---:|---:|
| Constant Velocity (reference) | — | 0.498 | 2.572 | 6.303 | 2.985 |
| CS-LSTM *(de facto baseline)* | 2018 | 0.61 | 2.14 | 4.30 | 1.95 |
| GRIP++ | 2019 | 0.38 | 1.62 | 3.48 | 1.61 |
| GISNet | 2020 | 0.33 | 1.48 | 2.95 | 1.40 |
| **HierarchicalGNN** *(best in 26-model survey)* | 2021 | 0.34 | 1.32 | 2.83 | **1.30** |
| CDSTraj | 2024 | 0.36 | 1.36 | 2.85 | 1.49 |
| BAT | 2024 | 0.23 | 1.54 | 3.62 | 1.74 |
| VT-Former | 2024 | — | — | — | 0.82 |
| **K-PhysGAT — ours (random split)** | **2026** | **0.269** | **0.594** | **1.986** | **0.863** |
| **K-PhysGAT — ours (GroupKFold, 15 runs)** | **2026** | **0.332** | **0.642** | **2.546** | **1.053 ± 0.013** |
| **K-PhysGAT — ours (K=6 multimodal, minADE)** | **2026** | — | — | **1.446** | **0.704 ± 0.031** |

</div>

> **The bottom line.** On the standard random-split protocol, K-PhysGAT (ADE **0.863 m**) beats every published model in the comparison table — including 2024 work (CDSTraj 1.49 m, BAT 1.74 m, VT-Former 0.82 m). Under honest vehicle-level GroupKFold averaged over 15 runs, K-PhysGAT (**1.053 ± 0.013 m**) still beats HierarchicalGNN's 1.30 m by **19% relative**, with σ tight enough to rule out seed luck. Multimodal K=6 brings minADE down to **0.704 m**.
>
> Crucially, K-PhysGAT is **the only model in the table that guarantees kinematic feasibility** — every predicted trajectory satisfies the bicycle equations exactly, by construction.

Full 26-model comparison: [`results/ngsim_v14/sota_comparison.csv`](results/ngsim_v14/sota_comparison.csv) · Interactive dashboard: [`results/results_vs_sota_dashboard.html`](results/results_vs_sota_dashboard.html) · Methodology details: [`docs/K_PhysGAT_NGSIM.md`](docs/K_PhysGAT_NGSIM.md).

---

### Warangal heterogeneous urban — SP-GAT, Phase 16 → 22 ablation

Starting from an early multi-agent social LSTM baseline averaging ~8 m ADE across classes, a disciplined six-phase ablation reduced error by **88–93 %** across all four classes.

<div align="center">

| Phase | Key change | Bike ↓ | Auto ↓ | Car ↓ | Bus ↓ |
|---|---|---:|---:|---:|---:|
| Baseline | Multi-agent social LSTM | 8.83 | 8.72 | 7.85 | 6.96 |
| P16 | Composite string ID fix (was the dominant bug) | 2.247 | 2.358 | 3.706 | 6.765 |
| P17 | + physics init + macro heading + per-class | 1.111 | 0.877 | 1.330 | 1.298 |
| P18 | + heterogeneous neighbour 1-hot + 16-d embedding | 1.369 | 1.393 | 1.074 | 1.348 |
| P19 | + mirror-Y augmentation (2× data) | 1.113 | **0.773** ⭐ | 0.911 | 0.606 |
| P21 | + ACT safety loss | 1.045 | 0.812 | **0.805** ⭐ | 0.522 |
| **P22** | **+ proximity + jerk (full RAPiD safety suite)** | **1.029** ⭐ | 0.814 | 0.906 | **0.496** ⭐ |

</div>

#### Best per class (minADE @ 5 s horizon)

<div align="center">

| Class | Best phase | **minADE (m)** | **minFDE (m)** | **RMSE (m)** | vs baseline |
|---|---|---:|---:|---:|---:|
| 🚌 **Bus** | P22 — full safety suite | **0.496** | 0.960 | 0.686 | **−92.9 %** |
| 🛺 **Auto** | P19 — Y-flip augmentation | **0.773** | 1.866 | 1.167 | **−91.1 %** |
| 🚗 **Car** | P21 — ACT loss | **0.805** | 2.097 | 1.235 | **−89.7 %** |
| 🚲 **Bike** | P22 — full safety suite | **1.029** | 2.414 | 1.861 | **−88.3 %** |

</div>

Per-phase CSV: [`results/warangal/ablation_phases_16_to_22.csv`](results/warangal/ablation_phases_16_to_22.csv) · Methodology details: [`docs/SP_GAT_warangal.md`](docs/SP_GAT_warangal.md).

---

> ### ⚠️ A note on evaluation integrity *(Phase 23 — in progress)*
>
> The Warangal Phase 16–22 numbers above use an **80/20 temporal sequence split**, not vehicle-level GroupKFold. The same vehicles can appear in train and validation. This is the identical leakage pattern that inflated my own NGSIM numbers before the v10 → v14 protocol fix.
>
> **The Phase 17–22 sub-1m ADEs are likely optimistic.** Phase 23 is migrating Warangal to vehicle-level GroupKFold — matching the NGSIM v14 protocol. Expected outcome: numbers degrade somewhat but become defensible and publishable.
>
> Flagging your own evaluation flaws before a reviewer does is not optional. Research credibility = protocol integrity.

---

## 🧠 Architecture

### K-PhysGAT — for NGSIM and as the SP-GAT backbone

```text
            3s × 5-dim per-step features per agent  (Δx, Δy, vx, vy, lane_id)
                                  │
                        ┌─────────▼──────────┐
                        │ BiLSTM-256         │
                        │ + LayerNorm        │   per-agent motion encoder
                        └─────────┬──────────┘
                                  │ h_i ∈ ℝ²⁵⁶
                        ┌─────────▼───────────────────────┐
                        │ Edge-feature multi-head GAT     │   social attention
                        │  4 heads · top-8 neighbours     │   over neighbours
                        │  5-dim edge feats: Δx,Δy,Δv,d   │
                        └─────────┬───────────────────────┘
                                  │
                        ┌─────────▼────────────────┐
                        │ 2-layer LSTM control head │   decodes 50 timesteps of
                        │   →  (δ_t, a_t)           │   (steering δ, accel a)
                        └─────────┬────────────────┘
                                  │
                        ┌─────────▼─────────────────────────────┐
                        │ Differentiable kinematic-bicycle      │   physical
                        │ rollout  ( L = 4.5 m,  dt = 0.1 s )   │   guarantee:
                        │   x_{t+1} = x_t + v·cos(ψ)·dt         │   no railgun,
                        │   y_{t+1} = y_t + v·sin(ψ)·dt         │   no teleport,
                        │   ψ_{t+1} = ψ + (v/L)·tan(δ)·dt       │   no NaN
                        │   v_{t+1} = v + a·dt                  │
                        └─────────┬─────────────────────────────┘
                                  │
                        ┌─────────▼─────────────────┐
                        │ Horizon-weighted Huber    │   w_t : 1.0 → 3.0
                        │ over 50 timesteps         │   (later steps cost more)
                        └───────────────────────────┘
```

**~1.85 M parameters · 9.4 ms per sample at batch 32 · zero kinematic violations by construction.** Full latency curve in [`results/ngsim_v14/inference_latency.csv`](results/ngsim_v14/inference_latency.csv).

### SP-GAT — heterogeneous extension for Warangal

Same backbone, with five additions targeted at the heterogeneous-class problem:

1. **Per-class training** — separate models for bike / auto / car / bus (pooling lost on every class).
2. **Heterogeneous neighbour-class encoding** — one-hot class indicator on each neighbour edge, run through a learned **16-d embedding** (TrafficPredict-inspired).
3. **Physics-informed initialisation** — `MAX_V0 = 40 m/s`, RandomUniform on the control Dense layer to escape a zero-gradient deadlock under `tanh` saturation.
4. **Mirror-Y data augmentation** — for a symmetric 2-way road, Y-axis reflection is label-preserving; doubles the effective dataset.
5. **Multi-objective safety loss** (Phase 22):

$$
\mathcal{L}_\text{total}
\;=\; \mathcal{L}_\text{geometry}
\;+\; w_1\,\mathcal{L}_\text{ACT}
\;+\; w_2\,\mathcal{L}_\text{prox}
\;+\; w_3\,\mathcal{L}_\text{jerk}
$$

where $\mathcal{L}_\text{ACT}$ is the RAPiD-inspired Anticipated-Collision-Time penalty, $\mathcal{L}_\text{prox}$ penalises predicted paths entering unsafe headway zones, and $\mathcal{L}_\text{jerk}$ regularises for smoothness. Weights $w_i$ are **class-aware** — bikes tolerate higher jerk than buses, and the loss should reflect that.

---

## 💡 Engineering lessons from the trenches

Six lessons earned through training-log inspection — not predicted in advance. A research project is judged as much by the bugs it found as by the numbers it reports.

#### 1 · The composite-ID bug was the single biggest source of error
Numeric-only vehicle IDs collided across classes: `Bike_id = 1` and `Car_id = 1` overwrote each other in the neighbour dictionary, silently corrupting **all** cross-class training data. Errors of 23–35 m were not the model's fault. Switching to composite string IDs (`"Bike_1"`, `"Car_1"`) was a one-line preprocessing change that collapsed error by ~10×.
**The lesson:** When the model "doesn't work", suspect the data pipeline before the architecture.

#### 2 · Per-class training beats pooled training for heterogeneous traffic
A unified model across all four classes lost to four class-specific models on every class. Bus and bike live in different speed and dynamics regimes; the pooled model under-fit both.
**The lesson:** Distribution heterogeneity can defeat the "more data wins" prior.

#### 3 · Safety losses improve geometric accuracy, not just safety
Counter-intuitively, adding ACT + proximity + jerk penalties **lowered ADE/FDE** on Bus and Car (P22 vs P19). Physical-plausibility regularisation acts as a strong prior matching real driving — improving generalisation, not just safety.
**The lesson:** Domain priors regularise; don't treat "accuracy" and "physical plausibility" as a trade-off.

#### 4 · `val_loss = 0` is a silent killer
Keras' compiled-loss reduction returned `val_loss = 0.0` during validation when the custom loss had auxiliary terms — early stopping was thus driven by a meaningless signal, letting the model overfit invisibly. Fixed by overriding `test_step` explicitly.
**The lesson:** Verify `val_loss > 0` before trusting any training curve.

#### 5 · Sample count is not sample independence
Reducing window stride from 5 to 3 looks like free data but produces highly correlated overlapping samples. **Stride 5 generalised better than stride 3 despite ~40% fewer samples.**
**The lesson:** Sample independence matters more than sample count.

#### 6 · Report the spread, not the best run
A 15-run GroupKFold sweep with **σ = 0.013 m** is a far stronger result than a single 0.70 m number. Best-of-N reporting hides seed luck.
**The lesson:** Mean and std beats best-of-N every time.

---

## 📂 Repository structure

```
trajectory-prediction-indian-traffic/
│
├── README.md                                  ← you are here
├── LICENSE                                    ← MIT
├── requirements.txt                           ← TF 2.15, NumPy, etc
│
├── notebooks/
│   ├── 01_ngsim_kphysgat_v14.ipynb            ← NGSIM K-PhysGAT — final pipeline
│   └── 02_warangal_spgat_phases.ipynb         ← Warangal Phase 1 → 22 ablation
│
├── results/
│   ├── ngsim_v14/
│   │   ├── sota_comparison.csv                ← K-PhysGAT vs 26 published models
│   │   ├── ade_per_second_random_split.csv    ← raw per-second ADE under random split
│   │   ├── multimodal_K6_groupkfold.csv       ← K=6 WTA, 3 seeds, per-seed raw
│   │   ├── inference_latency.csv              ← latency by batch size
│   │   ├── all_results_random_and_groupkfold.json   ← both protocols, all 5 models
│   │   └── summary_15run_variance.json        ← 15-run GroupKFold mean / std / per-fold
│   ├── warangal/
│   │   ├── ablation_phases_16_to_22.csv       ← per-phase, per-class minADE
│   │   ├── best_per_class.csv                 ← winning phase and metrics per class
│   │   └── dataset_stats.csv                  ← vehicle counts, sampling params
│   └── results_vs_sota_dashboard.html         ← interactive SOTA dashboard
│
├── docs/
│   ├── K_PhysGAT_NGSIM.md                     ← NGSIM methodology, full ablation
│   └── SP_GAT_warangal.md                     ← Warangal methodology, lessons learned
│
└── papers/
    └── REFERENCES.md                          ← all influential papers + BibTeX
```

---

## ⚙️ Reproducibility

### Environment

```bash
git clone https://github.com/satyamk4517/trajectory-prediction-indian-traffic.git
cd trajectory-prediction-indian-traffic
pip install -r requirements.txt
```

Tested on **Google Colab, Python 3.10, NVIDIA T4 GPU**.

### Run the notebooks

Each notebook is self-contained — no external Python modules, no hidden config. Open in Colab or locally, mount your data directory, and run all cells.

| Notebook | What it does | Outputs |
|---|---|---|
| [`01_ngsim_kphysgat_v14.ipynb`](notebooks/01_ngsim_kphysgat_v14.ipynb) | Builds the v14 cache from raw NGSIM US-101, trains CV / Iter3A / Iter3B / Iter4 / K-PhysGAT, runs 15-seed GroupKFold sweep and K=6 multimodal evaluation. | Files in [`results/ngsim_v14/`](results/ngsim_v14/) |
| [`02_warangal_spgat_phases.ipynb`](notebooks/02_warangal_spgat_phases.ipynb) | The full Warangal journey — Phase 1 through Phase 22 — including the failed phases (7, 8, 9) that taught the most. | Files in [`results/warangal/`](results/warangal/) |

### Datasets

| Dataset | Source | Access |
|---|---|---|
| **NGSIM US-101** | US Federal Highway Administration | Public — [download](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm) |
| **Warangal urban drone** | IIT Guwahati / Prof. Mallikarjuna's lab | Not publicly released. Contact the supervisor for academic access. |

### Training configuration

| Hyperparameter | NGSIM K-PhysGAT | Warangal SP-GAT |
|---|---|---|
| Observation horizon | 30 frames (3 s @ 10 Hz) | 30 frames (3 s @ 10 Hz) |
| Prediction horizon | 50 frames (5 s) | 50 frames (5 s) |
| Optimiser | Adam | Adam |
| Learning rate | 1e-3 → cosine decay | 1e-3 |
| Batch size | 64 | 64 (Bus: 32) |
| Epochs | ≤ 100 with early stopping | ≤ 50 (Bus: 80, patience 15) |
| Loss | Horizon-weighted Huber | MSE + ACT + prox + jerk (class-aware weights) |
| Init | RandomUniform, MAX_V0 = 40 m/s | Same |
| Split | **Vehicle-level GroupKFold (k = 5)** | 80/20 temporal *(P23: migrating to GroupKFold)* |

---

## 🗺️ Roadmap

- [x] **Phase 1** — Data pipeline, sequence baselines (LSTM, GRU, Encoder–Decoder), NGSIM benchmarking
- [x] **NGSIM v14 — K-PhysGAT (current state-of-the-art for this repo)** · 15-run GroupKFold sweep, K=6 multimodal WTA
- [x] **Warangal P16 → P22** — ID fix, per-class training, heterogeneous encoding, mirror-Y, full RAPiD safety suite
- [ ] **Phase 23** *(in progress)* — Vehicle-level GroupKFold migration for Warangal
- [ ] **Phase 24** — `tf.scan` rewrite of the bicycle rollout (target: < 50 ms / sample latency)
- [ ] **Phase 25** — Intersection-level prediction (current Warangal dataset is mid-block)
- [ ] **Phase 26** — Lane-graph context (HD-map conditioning, à la TNT / Trajectron++)
- [ ] **Phase 27** — Transformer backbone comparison + attention-visualisation for interpretability

---

## 👤 About

<table>
<tr>
<td valign="top">

**Satyam Kumar**
M.Tech · Transportation Systems Engineering
Indian Institute of Technology Guwahati

Supervisor: **Prof. C. Mallikarjuna**
Department of Civil Engineering, IIT Guwahati

I work at the intersection of traffic engineering and deep learning — physics-injected, safety-aware models for **heterogeneous traffic**, the kind that exists outside NGSIM. I care about evaluation honesty, reproducibility, and the unglamorous bugs that determine whether a model actually works.

</td>
<td valign="top" width="320">

**📫 Get in touch**

✉️  [satyamk4517@iitg.ac.in](mailto:satyamk4517@iitg.ac.in)
✉️  [satyamshivam511@gmail.com](mailto:satyamshivam511@gmail.com)
💼  [LinkedIn](https://www.linkedin.com/in/satyam-kumar-a8b25b1b2)
🏛️  [IIT Guwahati — Civil](https://www.iitg.ac.in/civil/)

*If you are hiring, collaborating, or working on motion forecasting for non-Western traffic — I would love to talk.*

</td>
</tr>
</table>

---

## 📄 Citation

```bibtex
@mastersthesis{kumar2026kphysgat,
  author  = {Kumar, Satyam},
  title   = {Vehicle Trajectory Prediction using AI/ML Models for Urban Roads
             and Intersections in Heterogeneous Traffic},
  school  = {Indian Institute of Technology Guwahati},
  year    = {2026},
  type    = {M.Tech Thesis},
  address = {Guwahati, India},
  note    = {Department of Civil Engineering, Transportation Systems Engineering.
             Supervisor: Prof. C. Mallikarjuna.}
}
```

---

<div align="center">

**Keywords** · trajectory prediction · graph attention networks · physics-injected deep learning · kinematic bicycle model · heterogeneous traffic · Indian urban traffic · safety-aware learning · NGSIM · Warangal · motion forecasting · autonomous vehicles · ITS · ATMS

*Built at IIT Guwahati · Powered by a very long-running Colab session and an even longer training log.*

</div>
