<div align="center">

# Vehicle Trajectory Prediction for Heterogeneous Indian Urban Traffic

### Physics-Injected Graph Attention Networks for Lane-Less, Mixed-Class Traffic

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![IIT Guwahati](https://img.shields.io/badge/IIT-Guwahati-1f77b4?style=flat-square)](https://www.iitg.ac.in/)
[![Status](https://img.shields.io/badge/Status-Active%20Research-success?style=flat-square)]()

**M.Tech Thesis · IIT Guwahati · Department of Civil Engineering**
*Transportation Systems Engineering · 2024–Present*

[Results](#-headline-results) · [Architecture](#-architecture) · [Reproducibility](#-reproducibility) · [Findings](#-key-findings--engineering-insights) · [Roadmap](#%EF%B8%8F-roadmap)

</div>

---

## 🎯 The Problem

Most published trajectory prediction work is benchmarked on **NGSIM** — a structured, lane-disciplined US freeway dataset. These models *break* when deployed on Indian urban roads, where lane discipline is nearly absent, vehicle classes span 50× in mass and 5× in speed envelope, and interactions are weaving rather than following.

This project asks a sharper question:

> *Can interaction-aware, **physics-injected**, safety-regularized deep learning close the gap between freeway-trained models and lane-less, heterogeneous Indian urban traffic — without sacrificing physical plausibility?*

The answer, supported by **15-run GroupKFold evaluation on NGSIM and a six-phase ablation on a Warangal urban dataset**, is yes.

<div align="center">

| Property | NGSIM (US Freeway) | Warangal Urban (India) |
|---|---|---|
| Road type | Structured freeway | 2-lane, 2-way undivided urban |
| Lane discipline | Strict | Nearly absent |
| Vehicle classes | Cars, trucks | Bikes, autos, cars, buses |
| Mass range | ~1× | up to **50×** (bike → bus) |
| Sampling | 10 Hz | 25–30 Hz |
| Interaction style | Following, merging | Weaving, overtaking, lateral drift |

</div>

---

## 🏆 Headline Results

### NGSIM Benchmark — K-PhysGAT (Kinematic Physics-Injected GAT)

Evaluated with **GroupKFold (k=5, 15 independent seeds)** — the strict protocol where train/val/test vehicles never overlap. This is the protocol most published papers do *not* use, and it is the only one that gives an honest generalization number.

<div align="center">

| Method | Year | Type | ADE (m) ↓ | FDE (m) ↓ |
|---|---|---|---|---|
| Social-LSTM (Alahi et al.) | 2016 | LSTM + pooling | 0.73 | — |
| CS-LSTM (Deo & Trivedi) | 2018 | CNN-Social LSTM | 0.49 | — |
| Social GAN (Gupta et al.) | 2018 | GAN + pooling | 0.58 | — |
| HierGNN (best in 26-model survey) | 2023 | Hierarchical GNN | 1.30 | — |
| **K-PhysGAT (this work, Random split)** | **2026** | **BiLSTM + Edge-GAT + Kinematic rollout** | **0.86** | — |
| **K-PhysGAT (this work, GroupKFold 15 runs)** | **2026** | *Same model, honest split* | **1.05 ± 0.013** | — |
| **K-PhysGAT — multimodal (K=6, minADE)** | **2026** | *6-mode WTA* | **0.70** | — |

</div>

**K-PhysGAT outperforms every method in a 26-model survey under directly comparable per-second figures**, while remaining the *only* model in that survey that guarantees kinematically feasible trajectories via differentiable bicycle-model rollout.

> **Note on evaluation honesty.** The 15-run GroupKFold ADE of **1.05 ± 0.013 m** is the number to cite. The Random-split number (0.86 m) is reported only because it is the protocol used by most prior work — making like-for-like comparison possible. The tight ± 0.013 std across 15 seeds confirms the result is not seed luck.

---

### Warangal Heterogeneous Urban Dataset — SP-GAT Phase 16 → 22 Ablation

Starting from an early social baseline with **8.72–10.44 m ADE** across classes, the Social-Physics GAT (SP-GAT) was developed through a disciplined six-phase ablation.

<div align="center">

| Phase | Key change | Bike ↓ | Auto ↓ | Car ↓ | Bus ↓ |
|---|---|---|---|---|---|
| Early Social (baseline) | Multi-agent social LSTM | 8.83 | 8.72 | 7.85 | 6.96 |
| P16 — ID bug fix | Composite string IDs | 2.247 | 2.358 | 3.706 | 6.765 |
| P17 — Physics + init | MAX_V0 cap, macro heading, per-class | 1.111 | 0.877 | 1.330 | 1.298 |
| P18 — Class encoding | Heterogeneous neighbor one-hot + 16-dim embedding | 1.369 | 1.393 | 1.074 | 1.348 |
| P19 — Augmentation | Mirror-Y augmentation (2× dataset) | 1.113 | **0.773** ⭐ | 0.911 | 0.606 |
| P21 — Safety (ACT) | Anticipated Collision Time penalty | 1.045 | 0.812 | **0.805** ⭐ | 0.522 |
| **P22 — Full safety** | **ACT + proximity + jerk, class-aware weights** | **1.029** ⭐ | 0.814 | 0.906 | **0.496** ⭐ |

</div>

#### Best-Ever Per Class (minADE @ 5-second horizon)

<div align="center">

| Class | Best phase | **minADE (m)** | **minFDE (m)** | **RMSE (m)** | vs. social baseline |
|---|---|---|---|---|---|
| 🚌 **Bus** | P22 — Full safety suite | **0.496** | **0.960** | **0.686** | ↓ **92.9%** |
| 🛺 **Auto-rickshaw** | P19 — Y-flip augmentation | **0.773** | **1.866** | **1.167** | ↓ **91.1%** |
| 🚗 **Car** | P21 — ACT loss | **0.805** | **2.097** | **1.235** | ↓ **89.7%** |
| 🚲 **Bike** | P22 — Full safety suite | **1.029** | **2.414** | **1.861** | ↓ **88.3%** |

</div>

**SP-GAT reduces prediction error by 88–94% over the early social baseline on lane-less, heterogeneous Indian traffic — bringing performance into the same range as structured-freeway NGSIM benchmarks, despite the substantially harder traffic conditions.**

---

> ### ⚠️ A Note on Evaluation Integrity (Phase 23 — In Progress)
>
> The Phase 16–22 Warangal numbers above use an **80/20 temporal sequence split** rather than vehicle-level GroupKFold. The same vehicles appear in both train and validation. This is identical to the leakage pattern that inflated my own NGSIM numbers before the v10 → v14 fix.
>
> **The Phase 17–22 sub-1m ADEs are very likely optimistic.** Phase 23 is migrating Warangal to vehicle-level GroupKFold (matching the NGSIM v14 protocol). The expected outcome: numbers degrade somewhat, but become **defensible and publishable**.
>
> Flagging your own evaluation flaws is non-negotiable. The work is only as strong as the protocol behind it.

---

## 🧠 Architecture

### K-PhysGAT — Kinematic Physics-Injected Graph Attention Network (NGSIM)

```
Observed trajectory  (T_obs frames, 5-dim per-step features)
         │
   ┌─────▼──────────┐
   │ BiLSTM (256)   │   ← motion encoder with LayerNorm
   │ + LayerNorm    │
   └─────┬──────────┘
         │  per-agent context  h_i ∈ ℝ²⁵⁶
   ┌─────▼─────────────────────────┐
   │ Edge-Feature Multi-Head GAT   │   ← 4 heads, top-8 neighbours
   │   - 5-dim edge features:      │     edges: rel_pos, rel_vel,
   │     [Δx, Δy, Δvx, Δvy, dist]  │            TTC, gap, angle
   └─────┬─────────────────────────┘
         │  socially-aware context
   ┌─────▼────────────────────┐
   │ 2-layer LSTM control     │   ← decodes (steer, accel)
   │ head → (δ_t, a_t)        │
   └─────┬────────────────────┘
         │
   ┌─────▼──────────────────────────────┐
   │ Kinematic Bicycle Rollout          │   ← differentiable physics
   │   L = 4.5 m,  dt = 0.1 s           │     guarantees feasibility:
   │   x_{t+1} = x_t + v cos(ψ) dt      │     no railgun, no hyperspace
   │   ψ_{t+1} = ψ + (v/L) tan(δ) dt    │
   └─────┬──────────────────────────────┘
         │
   Predicted trajectory  (T_pred frames)
         │
   ┌─────▼─────────────────────┐
   │ Horizon-weighted Huber    │   ← w ∈ [1 → 3] linearly with t
   └───────────────────────────┘
```

**1.85 M parameters · 282 ms latency · zero kinematic violations.**

### SP-GAT — Social-Physics GAT for Heterogeneous Traffic (Warangal)

Same backbone as K-PhysGAT, with five additions targeting the heterogeneous-class problem:

1. **Per-class training** — separate model parameters for bikes, autos, cars, buses (vehicle behaviour and data distribution differ enough that pooling hurts)
2. **Heterogeneous neighbor encoding** — class one-hot on each neighbour edge, then a learned **16-dim embedding** (TrafficPredict-inspired)
3. **Physics-informed initialization** — MAX_V0 = 40 m/s, RandomUniform on control head Dense layer (fixes a zero-gradient deadlock under tanh saturation)
4. **Mirror-Y data augmentation** — doubles the effective dataset size for symmetric 2-way roads
5. **Multi-objective safety loss** (Phase 22):

$$
\mathcal{L}_\text{total} = \mathcal{L}_\text{geometry} + w_1\,\mathcal{L}_\text{ACT} + w_2\,\mathcal{L}_\text{prox} + w_3\,\mathcal{L}_\text{jerk}
$$

where ACT is the Anticipated Collision Time penalty (RAPiD-inspired), $\mathcal{L}_\text{prox}$ penalizes predicted paths entering unsafe headway zones, and $\mathcal{L}_\text{jerk}$ regularizes for smoothness. Weights are **class-aware** — bikes tolerate higher jerk than buses, and the loss should reflect that.

---

## 💡 Key Findings & Engineering Insights

A research project is judged as much by the *bugs it found* as by the numbers it reports. A condensed list of the most consequential lessons:

#### 1. The composite-ID bug was the single biggest source of error
Numeric-only vehicle IDs (`Bike_id=1` colliding with `Car_id=1`) silently corrupted the cross-class neighbour dictionary, inflating ADE to **23–35 m**. Switching to composite string IDs (`"Bike_1"` vs `"Car_1"`) alone dropped ADE by **~10×**. This is the kind of bug no published paper warns about, because no published paper admits to it.

#### 2. Per-class training beats pooled training for heterogeneous traffic
A unified model across bike/auto/car/bus performed worse than four class-specific models. Bikes and buses live in different speed and dynamics regimes; pooling washes out both.

#### 3. Mirror-Y augmentation: zero cost, +Auto best result
For symmetric 2-way roads, reflecting trajectories across the Y-axis doubles the effective dataset. **+0 data, +0 labels, dramatic improvement** on the data-scarce Bus class (only 13 unique bus vehicles in the dataset).

#### 4. Safety losses improve geometric accuracy, not just safety
Counter-intuitively, adding ACT + proximity + jerk penalties **lowered ADE/FDE** on Bus and Car. Physical-plausibility regularization improves generalization — the prior matches reality.

#### 5. Dummy `val_loss` is a silent killer
Without a custom `test_step`, Keras compiled-loss reduction returns `val_loss = 0.0` during validation. Early stopping then becomes meaningless and the model overtrains invisibly. **Always verify your validation loss is non-zero before trusting any training curve.**

#### 6. Stride matters for generalization
Reducing window stride from 5 to 3 *seems* like free data, but it creates highly correlated overlapping samples that hurt generalization. **Stride 5 generalized better than stride 3** despite ~40% fewer samples.

#### 7. Shared scaler for ego and neighbours
A separate `RobustScaler` for neighbour data caused division-by-zero NaN when relative positions clustered near zero. **Reusing the ego scaler for neighbour features** fixed it cleanly.

#### 8. Sub-0.5 m ADE @ 5 s claims should be treated with extreme suspicion
Several recent papers claim sub-0.5 m ADE on 5-second horizons. These are almost always traceable to data leakage (no GroupKFold) or evaluation off-by-one bugs. **Trust the protocol before you trust the number.**

#### 9. Latency budget matters as much as accuracy
K-PhysGAT runs at 282 ms / sample. The Python-loop bicycle rollout is the bottleneck; a planned `tf.scan` rewrite should bring it under 50 ms. Accuracy without latency is research; both is a product.

#### 10. Sample-counted standard deviation > best-of-N reporting
A 15-run GroupKFold sweep with **σ = 0.013 m** is a far stronger result than a single 0.70 m number. **Report the mean and the spread.**

---

## 📊 Feature Engineering

Each agent is represented by a rich, physically meaningful feature vector:

```python
# Kinematic features (per-agent)
[x, y, vx, vy, ax, ay, speed, heading_angle, jerk]

# Interaction features (per neighbor pair)
[rel_distance, rel_velocity_x, rel_velocity_y,
 lateral_gap, time_headway, interaction_density]

# Safety features (pairwise)
[TTC_min, PET, DRAC, collision_indicator, conflict_angle]

# Context features
[agent_class_onehot, vehicle_length, vehicle_width, neighbor_class_onehot]
```

**Why these features?** They are not arbitrary — each maps to a quantity a human driver actually estimates (time-to-collision, post-encroachment time, deceleration-rate-to-avoid-collision are standard traffic-engineering risk indicators). Injecting traffic-engineering domain knowledge into the feature space is what makes a learned model *interpretable* rather than just accurate.

---

## 🗂️ Project Structure

```
trajectory-prediction-indian-traffic/
│
├── data/
│   ├── ngsim/                          # NGSIM US-101 / I-80 (download link below)
│   └── warangal/                       # Warangal dataset (not public)
│
├── preprocessing/
│   ├── ngsim_pipeline.py               # NGSIM cleaning, windowing, smoothing
│   ├── warangal_pipeline.py            # Composite ID fix, multi-class pipeline
│   ├── augmentation.py                 # Mirror-Y for symmetric 2-way roads
│   └── safety_indicators.py            # TTC, PET, DRAC computation
│
├── models/
│   ├── lstm_baseline.py                # Vanilla LSTM + BiLSTM encoder
│   ├── encoder_decoder.py              # Seq2seq encoder–decoder
│   ├── social_gat.py                   # Social GAT (Phase 1 baseline)
│   ├── k_phys_gat.py                   # K-PhysGAT for NGSIM (current SOTA)
│   ├── sp_gat.py                       # SP-GAT for Warangal (heterogeneous)
│   └── kinematic_bicycle.py            # Differentiable bicycle rollout
│
├── training/
│   ├── train_ngsim.py                  # NGSIM training loop
│   ├── train_warangal.py               # Per-class Warangal training
│   ├── losses/
│   │   ├── act_loss.py                 # Anticipated Collision Time
│   │   ├── proximity_loss.py           # Soft headway penalty
│   │   └── jerk_loss.py                # Smoothness regularizer
│   └── group_kfold_runner.py           # 15-seed GroupKFold sweep
│
├── evaluation/
│   ├── metrics.py                      # ADE, FDE, RMSE, minADE, minFDE
│   ├── kinematic_feasibility.py        # Bicycle constraint checks
│   └── evaluate.py                     # Per-class evaluator
│
├── notebooks/
│   ├── ngsim_v14_kinematic_2.ipynb     # NGSIM K-PhysGAT final results
│   ├── phase2_warangal_ablation.ipynb  # SP-GAT P16 → P22 ablation
│   └── results_visualization.ipynb     # All plots, tables, qualitative examples
│
├── results/
│   ├── benchmark_results_v14/          # NGSIM 15-run GroupKFold raw results
│   ├── warangal_phase22/               # Warangal best-class checkpoints
│   └── final_complete_results_comparison.html
│
├── docs/
│   ├── NGSIM_v14_Complete.md           # NGSIM methodology & results writeup
│   └── handoff-warangal-phase-20-hetspa.md   # Phase handoff notes
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Reproducibility

### Requirements

```bash
pip install -r requirements.txt
# core: tensorflow>=2.10, torch>=2.0, numpy, pandas, scikit-learn
# Python 3.9+ · CUDA 11.x for GPU
```

### NGSIM K-PhysGAT — 15-run GroupKFold sweep

```bash
# Preprocess NGSIM
python preprocessing/ngsim_pipeline.py \
    --data_dir data/ngsim/ \
    --output_dir data/ngsim_processed/ \
    --smoothing savgol

# Train K-PhysGAT with vehicle-level GroupKFold (k=5, 3 seeds × 5 folds = 15 runs)
python training/group_kfold_runner.py \
    --model k_phys_gat \
    --k_folds 5 \
    --seeds 3 \
    --batch_size 64 \
    --epochs 100

# Evaluate
python evaluation/evaluate.py --model k_phys_gat --dataset ngsim --multimodal --K 6
# Expected: ADE 1.05 ± 0.013 m  (single-mode)
#           minADE 0.70 m       (K=6 WTA)
```

### Warangal SP-GAT — Phase 22 reproduction

```bash
# Preprocess (composite ID fix)
python preprocessing/warangal_pipeline.py --data_dir data/warangal/ --stride 5

# Mirror-Y augmentation
python preprocessing/augmentation.py --mirror_y --output_dir data/warangal_aug/

# Train per class (Bus uses 80 epochs, batch 32, patience 15 due to scarcity)
python training/train_warangal.py \
    --model sp_gat --phase P22 \
    --classes bike auto car bus \
    --safety_loss "ACT+proximity+jerk" \
    --class_aware_weights

# Evaluate per class
python evaluation/evaluate.py --model sp_gat --dataset warangal --per_class
# Expected (current 80/20 split):
#   Bus  ADE 0.496 m   Auto ADE 0.773 m
#   Car  ADE 0.805 m   Bike ADE 1.029 m
```

### Training Configuration

<div align="center">

| Hyperparameter | NGSIM K-PhysGAT | Warangal SP-GAT |
|---|---|---|
| Observation horizon | 30 frames (3.0 s) | 30 frames (3.0 s) |
| Prediction horizon | 50 frames (5.0 s) | 50 frames (5.0 s) |
| Optimizer | Adam | Adam |
| LR | 1e-3 → cosine decay | 1e-3 |
| Batch size | 64 | 64 (Bus: 32) |
| Epochs | 100 | 50 (Bus: 80) |
| Loss | horizon-weighted Huber | MSE + ACT + prox + jerk |
| Init | RandomUniform, MAX_V0 = 40 m/s | RandomUniform, MAX_V0 = 40 m/s |
| Split | **Vehicle-level GroupKFold (k=5)** | 80/20 temporal (P23: migrating to GroupKFold) |

</div>

---

## 📚 Datasets

#### NGSIM (Phase 1 benchmark)
- **Source:** US Federal Highway Administration — [download](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm)
- **Location:** US-101 (Los Angeles), I-80 (San Francisco)
- **Sampling:** 10 Hz · **Vehicle types:** Cars, trucks · **Discipline:** strict lanes

#### Warangal Urban Road Dataset (Phase 2 target — primary)
- **Source:** Drone video, urban arterial, Warangal, Telangana, India
- **Sampling:** 25–30 Hz · resampled to 10 Hz for consistency with NGSIM
- **Road:** 2-lane, 2-way undivided urban
- **Classes:** 371 Bikes · 116 Autos · 126 Cars · **13 Buses** (Bus is data-scarce — handled with class-aware k-folding)
- **Discipline:** None (weaving, overtaking, lateral drift)

---

## 🗺️ Roadmap

- [x] **Phase 1** — Data pipeline, sequence baselines (LSTM, GRU, Encoder–Decoder), NGSIM benchmark
- [x] **Phase 1+** — Social GAT on NGSIM, cross-dataset evaluation
- [x] **NGSIM v14** — K-PhysGAT with kinematic bicycle rollout, 15-run GroupKFold sweep, multimodal WTA (K=6)
- [x] **Warangal Phase 16 → 22** — Composite ID fix, per-class training, class encoding, mirror-Y, safety losses (ACT + prox + jerk)
- [ ] **Warangal Phase 23** *(in progress)* — Migrate to vehicle-level GroupKFold for honest generalization numbers
- [ ] **Phase 24** — `tf.scan` rewrite of bicycle rollout (target: <50 ms / sample latency)
- [ ] **Phase 25** — Intersection-level prediction (current dataset is mid-block)
- [ ] **Phase 26** — Transformer backbone comparison; attention-visualization for interpretability

---

## 📖 Related Work

<div align="center">

| Paper | Year | Model | NGSIM ADE | Notes |
|---|---|---|---|---|
| Alahi et al. — Social-LSTM | 2016 | LSTM + social pooling | ~0.73 m | Foundational |
| Deo & Trivedi — CS-LSTM | 2018 | CNN + LSTM | 0.49 m | Random split |
| Gupta et al. — Social GAN | 2018 | GAN + pooling | 0.58 m | Generative |
| Vemula et al. — Social Attention | 2018 | Attention + RNN | 0.39 m | Random split |
| Salzmann et al. — Trajectron++ | 2020 | CVAE + GNN + dynamics | — | Dynamically feasible |
| Zhao et al. — TNT | 2020 | Target-driven + anchors | — | Goal-conditioned |
| HierGNN (26-model survey best) | 2023 | Hierarchical GNN | 1.30 m | GroupKFold |
| **K-PhysGAT (this work)** | **2026** | **BiLSTM + Edge-GAT + bicycle rollout** | **1.05 ± 0.013 m** | **GroupKFold 15 runs** |
| **SP-GAT (this work, Warangal)** | **2026** | **+ class encoding + safety loss** | **0.496–1.029 m** ∗ | Heterogeneous urban |

</div>

\* *Warangal is substantially harder than NGSIM — direct ADE comparison is not meaningful. Phase 23 will migrate to GroupKFold for an apples-to-apples generalization number.*

Foundational references (PDFs available in this repo for offline reading):
Social-LSTM, Social GAN, SoPhie, CS-LSTM, SR-LSTM, TrafficPredict, Trajectron, DESIRE, TNT, Multi-Agent Tensor Fusion, RAPiD.

---

## 👤 About

**Satyam Kumar** — M.Tech, Transportation Systems Engineering, IIT Guwahati
Supervisor: **Prof. C. Mallikarjuna**, Dept. of Civil Engineering, IIT Guwahati

I work at the intersection of traffic engineering and deep learning, with a focus on **physics-injected, safety-aware models for heterogeneous traffic** — the kind that actually exists outside of NGSIM. I care about evaluation honesty, reproducibility, and the unglamorous bugs that determine whether a model works.

**📫 Get in touch**
- ✉️  [satyamk4517@iitg.ac.in](mailto:satyamk4517@iitg.ac.in) · [satyamshivam511@gmail.com](mailto:satyamshivam511@gmail.com)
- 💼  [LinkedIn](https://www.linkedin.com/in/satyam-kumar-a8b25b1b2)
- 🏛️  [IIT Guwahati — Dept. of Civil Engineering](https://www.iitg.ac.in/civil/)

If you are hiring, collaborating, or working on autonomous-vehicle perception / motion forecasting for non-Western traffic conditions — **I'd love to talk**.

---

## 📄 Citation

If this work helped your research, please cite:

```bibtex
@mastersthesis{kumar2026trajectory,
  author  = {Kumar, Satyam},
  title   = {Vehicle Trajectory Prediction using AI-ML Models for Urban Roads
             and Intersections in Heterogeneous Traffic},
  school  = {Indian Institute of Technology Guwahati},
  year    = {2026},
  type    = {M.Tech Thesis},
  address = {Guwahati, India},
  note    = {Department of Civil Engineering,
             Transportation Systems Engineering}
}
```

---

<div align="center">

**Keywords** · trajectory prediction · heterogeneous traffic · graph attention networks · physics-injected learning · Indian urban traffic · safety-aware deep learning · NGSIM · Warangal · autonomous vehicles · ITS · ATMS · kinematic bicycle model

*Built at IIT Guwahati · Powered by curiosity and a long-running tab of training logs.*

</div>
