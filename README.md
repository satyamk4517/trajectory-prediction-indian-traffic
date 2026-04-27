# Vehicle Trajectory Prediction for Heterogeneous Indian Urban Traffic

> **M.Tech Thesis — IIT Guwahati** | Dept. of Civil Engineering | Supervisor: Prof. C. Mallikarjuna  
> *Satyam Kumar (Roll No. 244104517) · Transportation Systems Engineering · 2024–Present*

---

## Why This Problem Is Hard

Trajectory prediction on **Indian urban roads** is a fundamentally different challenge from the standard NGSIM benchmark:

| Property | NGSIM (US Freeway) | Warangal Urban Road (India) |
|---|---|---|
| Road type | Structured freeway (I-101, SF) | 2-lane, 2-way undivided urban |
| Lane discipline | Strict | Nearly absent |
| Vehicle mix | Cars, trucks | Cars, bikes, autos, buses, trucks |
| Sampling rate | 10 Hz | 25–30 Hz |
| Traffic pattern | Predictable, homogeneous | Chaotic, heterogeneous |
| Interaction style | Following, merging | Weaving, overtaking, lateral drift |

Most published trajectory models are trained and evaluated on NGSIM. **They break badly on Indian traffic.** This project asks: *can interaction-aware, safety-regularized deep learning close that gap?*

---

## Key Results at a Glance

### NGSIM — Social GAT Benchmark (Phase 1)

| Model | Architecture | ADE (m) ↓ | FDE (m) ↓ |
|---|---|---|---|
| Kinematic LSTM (3A) | BiLSTM + kinematic decoder | 0.652 | 1.704 |
| Leader-Follower LSTM (3B) | BiLSTM + leader history (gated) | 0.613 | 1.569 |
| **Social GAT (4)** | **BiLSTM + Graph Attention** | **0.454** ⭐ | **1.102** ⭐ |

*Cross-validation best: Social GAT → ADE 0.454 m / FDE 1.102 m on NGSIM cars.*  
*Competitive with Deo & Trivedi (2018) CS-LSTM baseline.*

---

### Warangal — SP-GAT Iterative Development (Phase 2)

Starting from an early social baseline with **8.72–10.44 m ADE** across vehicle classes, the SP-GAT was developed through a disciplined ablation study:

| Phase | Key Change | Bike ADE ↓ | Auto ADE ↓ | Car ADE ↓ | Bus ADE ↓ |
|---|---|---|---|---|---|
| Early Social (baseline) | Multi-agent social LSTM | 8.83 m | 8.72 m | 7.85 m | 6.96 m |
| P16 (ID fix) | Composite string IDs (Bug fix) | 2.247 m | 2.358 m | 3.706 m | 6.765 m |
| P17 (Physics + init) | MAX_V0 cap, macro heading, per-class training | 1.111 m | 0.877 m | 1.330 m | 1.298 m |
| P18 (Class encoding) | Heterogeneous neighbor class one-hot | 1.369 m | 1.393 m | 1.074 m | 1.348 m |
| P19 (Augmentation) | Mirror-Y data augmentation (2× dataset) | 1.113 m | **0.773 m** ⭐ | 0.911 m | 0.606 m |
| P21 (ACT loss) | Anticipated Collision Time safety penalty | 1.045 m | 0.812 m | **0.805 m** ⭐ | 0.522 m |
| **P22 (Full safety)** | **ACT + proximity + jerk, class-aware thresholds** | **1.029 m** ⭐ | 0.814 m | 0.906 m | **0.496 m** ⭐ |

### Best-Ever Per Class (Warangal)

| Vehicle Class | Best Phase | **minADE (m)** | **minFDE (m)** | **RMSE (m)** | vs. Social Baseline |
|---|---|---|---|---|---|
| 🚌 Bus | P22 — Full Safety Suite | **0.496** | **0.960** | **0.686** | ↓ **94.5%** |
| 🛺 Auto-rickshaw | P19 — Y-Flip Augmentation | **0.773** | **1.866** | **1.167** | ↓ **91.1%** |
| 🚗 Car | P21 — ACT Loss | **0.805** | **2.097** | **1.235** | ↓ **91.1%** |
| 🚲 Bike | P22 — Full Safety Suite | **1.029** | **2.414** | **1.861** | ↓ **88.3%** |

> **Bottom line:** SP-GAT reduces prediction error by 88–94% over the social baseline on heterogeneous Indian traffic, bringing performance within the range of structured-freeway NGSIM benchmarks — despite the substantially harder traffic conditions.

---

## Architecture Overview

### Phase 1 — Sequence Baselines (NGSIM)

```
Observed trajectory (H frames)
        │
   ┌────▼─────┐
   │  BiLSTM  │  ← encodes motion history for each agent
   │  Encoder │
   └────┬─────┘
        │  context vector
   ┌────▼──────────────────┐
   │  Graph Attention Net  │  ← social interaction (Social GAT)
   │  (per-agent nodes,    │     edge features: relative pos, vel,
   │   interaction edges)  │     TTC, conflict angle
   └────┬──────────────────┘
        │
   ┌────▼─────┐
   │  LSTM    │  ← decodes future trajectory (F frames)
   │  Decoder │
   └────┬─────┘
        │
   Predicted trajectory (F frames) → ADE / FDE evaluation
```

### Phase 2 — SP-GAT (Safety-Penalty Graph Attention Network)

Extends the Social GAT with:

- **Per-class training** — separate model parameters for bikes, autos, cars, buses (vehicle behaviour differs fundamentally)
- **Heterogeneous neighbor encoding** — class one-hot on neighbor nodes (a bike near a bus behaves differently than a car near a bus)
- **Physics-informed initialization** — velocity-bounded initialization (MAX_V0 = 40 m/s), macro heading angle
- **Mirror-Y data augmentation** — doubles effective dataset size for 2-way roads
- **Multi-objective safety loss:**

```
L_total = L_geometry + w₁·L_ACT + w₂·L_proximity + w₃·L_jerk

where:
  L_geometry  = ADE + FDE (trajectory accuracy)
  L_ACT       = Anticipated Collision Time penalty (collision-averse prediction)
  L_proximity = soft penalty for predicted paths entering unsafe headway zones
  L_jerk      = smoothness regularization (physically plausible trajectories)
  w₁,w₂,w₃   = class-aware weights tuned per vehicle type
```

---

## Datasets

### NGSIM (Phase 1 benchmark)
- **Source:** US Federal Highway Administration — [Download](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm)
- **Location:** US-101 (Los Angeles) and I-80 (San Francisco)
- **Sampling:** 10 Hz, 15 min per segment
- **Vehicle types:** Cars, trucks, buses (homogeneous, lane-disciplined)
- **Usage:** Baseline benchmarking of sequence models

### Warangal Urban Road Dataset (Phase 2 target)
- **Source:** High-altitude camera, urban arterial road, Warangal, Telangana, India
- **Sampling:** ~25–30 Hz
- **Road type:** 2-lane, 2-way undivided urban road
- **Vehicle types:** Cars, motorcycles, auto-rickshaws, buses (heterogeneous, no lane discipline)
- **Usage:** Primary evaluation dataset — the real challenge this project targets

---

## Feature Engineering

Each agent is represented by a rich feature vector:

**Kinematic features (per-agent):**
```
[x, y, vx, vy, ax, ay, speed, heading_angle, jerk]
```

**Interaction features (per neighbor pair):**
```
[rel_distance, rel_velocity_x, rel_velocity_y,
 lateral_gap, time_headway, interaction_density]
```

**Safety features (pairwise):**
```
[TTC_min, PET, DRAC, collision_indicator, min_safe_distance, conflict_angle]
```

**Context features:**
```
[agent_class_onehot, vehicle_length, vehicle_width, neighbor_class_onehot]
```

---

## Project Structure

```
trajectory-prediction-indian-traffic/
│
├── data/
│   ├── ngsim/                  # NGSIM preprocessed (not included — see download link)
│   └── warangal/               # Warangal dataset (not publicly available)
│
├── preprocessing/
│   ├── ngsim_pipeline.py       # NGSIM cleaning, windowing, feature extraction
│   ├── warangal_pipeline.py    # Warangal multi-class pipeline, composite ID fix
│   └── augmentation.py         # Mirror-Y augmentation for 2-way roads
│
├── models/
│   ├── lstm_baseline.py        # Vanilla LSTM + BiLSTM encoder
│   ├── encoder_decoder.py      # Sequence-to-sequence Encoder–Decoder LSTM
│   ├── social_gat.py           # Social Graph Attention Network (Phase 1)
│   └── sp_gat.py               # Safety-Penalty GAT — Phase 2 main model
│
├── training/
│   ├── train_ngsim.py          # Training loop for NGSIM baselines
│   ├── train_warangal.py       # Per-class training for Warangal
│   └── safety_loss.py          # ACT + proximity + jerk multi-objective loss
│
├── evaluation/
│   ├── metrics.py              # ADE, FDE, RMSE, minADE, minFDE, TTC, collision rate
│   └── evaluate.py             # Per-class evaluation runner
│
├── notebooks/
│   ├── phase1_ngsim_results.ipynb      # NGSIM baseline experiments
│   ├── phase2_warangal_ablation.ipynb  # SP-GAT ablation P16→P22
│   └── results_visualization.ipynb     # All plots and comparison tables
│
├── results/
│   └── final_complete_results_comparison.html  # Full results table
│
├── requirements.txt
└── README.md
```

---

## Reproducing Results

### Requirements
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
# Python 3.9+, PyTorch 2.0+
```

### Quick start — NGSIM Social GAT
```bash
# Preprocess NGSIM
python preprocessing/ngsim_pipeline.py --data_dir data/ngsim/ --output_dir data/ngsim_processed/

# Train Social GAT
python training/train_ngsim.py --model social_gat --epochs 100 --lr 1e-3

# Evaluate
python evaluation/evaluate.py --model social_gat --dataset ngsim
# Expected: ADE ≈ 0.454 m, FDE ≈ 1.102 m
```

### Quick start — SP-GAT on Warangal (Phase 2)
```bash
# Preprocess Warangal (composite ID fix included)
python preprocessing/warangal_pipeline.py --data_dir data/warangal/ --stride 3

# Apply Y-flip augmentation
python preprocessing/augmentation.py --mirror_y

# Train SP-GAT per class
python training/train_warangal.py --model sp_gat --phase P22 \
    --classes bike auto car bus \
    --safety_loss ACT+proximity+jerk

# Evaluate per class
python evaluation/evaluate.py --model sp_gat --dataset warangal --per_class
# Expected: Bus ADE ≈ 0.496 m, Auto ≈ 0.773 m, Car ≈ 0.805 m, Bike ≈ 1.029 m
```

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Observation horizon | 10–20 frames |
| Prediction horizon | 20–30 frames |
| Optimizer | Adam |
| Base learning rate | 1e-3 |
| Batch size | 64 |
| Epochs | 50–200 (early stopping on val loss) |
| Loss (Phase 1) | MSE (geometric only) |
| Loss (Phase 2, SP-GAT) | MSE + ACT + proximity + jerk (class-aware weights) |
| Initialization | RandomUniform, MAX_V0 = 40 m/s (physics-bounded) |
| Data split | 70% train / 15% val / 15% test |

---

## Ablation Insights

A few discoveries from the P16→P22 ablation that are worth calling out:

**1. The composite ID bug mattered enormously (P16).** When agent IDs were numeric (`Bike_id=1` colliding with `Car_id=1`), the model received corrupted cross-class identity data, causing ADE to explode to 23–34 m. Fixing to composite string IDs (`"Bike_1"` vs `"Car_1"`) alone dropped ADE by ~10×. This is a subtle but critical preprocessing pitfall for heterogeneous-traffic datasets.

**2. Per-class training outperformed pooled training (P17).** A single model trained on all vehicle types simultaneously performed worse than separate class-specific models. Bikes and buses have fundamentally different dynamics — pooling hurt both.

**3. Data augmentation via mirror-Y was high-ROI (P19).** For a 2-lane, 2-way road, mirroring trajectories across the Y-axis (reflecting the road symmetry) doubled the effective dataset size with zero additional data collection. This produced the best Auto result (0.773 m) and significantly improved all other classes.

**4. Safety loss improved accuracy, not just safety (P21–P22).** Counter-intuitively, adding ACT + proximity + jerk penalties improved geometric ADE/FDE on several classes while also producing safer predicted trajectories. Regularizing for physical plausibility appears to help generalization.

---

## Phase Roadmap

- [x] **Phase 1** — Data pipeline, sequence baselines (LSTM, GRU, Encoder–Decoder), NGSIM benchmarking
- [x] **Phase 1+** — Social GAT on NGSIM, transfer to Warangal, cross-dataset evaluation
- [x] **Phase 2** — SP-GAT iterative development (P16→P22), per-class training, safety-aware loss
- [ ] **Phase 3** *(upcoming)* — Transformer-based trajectory prediction, attention visualization
- [ ] **Phase 3** *(upcoming)* — GNN extensions with edge-attention for conflict prediction
- [ ] **Phase 3** *(upcoming)* — Intersection-level trajectory prediction (current dataset is mid-block)

---

## Related Work

| Paper | Year | Model | NGSIM ADE |
|---|---|---|---|
| Alahi et al. — Social-LSTM | 2016 | LSTM + social pooling | ~0.73 m |
| Deo & Trivedi — CS-LSTM | 2018 | CNN + LSTM | 0.49 m |
| Gupta et al. — Social GAN | 2018 | GAN + pooling | 0.58 m |
| Vemula et al. — Social Attention | 2018 | Attention + RNN | 0.39 m |
| **This work — Social GAT** | **2025** | **BiLSTM + GAT** | **0.454 m** |
| **This work — SP-GAT (Warangal)** | **2025** | **GAT + safety loss** | **0.496–1.029 m*** |

*\*Warangal heterogeneous traffic is substantially harder than NGSIM — direct comparison not meaningful.*

---

## About

**Satyam Kumar** — M.Tech Transportation Systems Engineering, IIT Guwahati  
- 📧 satyamk4517@iitg.ac.in · satyamshivam511@gmail.com  
- 🔗 [LinkedIn](https://www.linkedin.com/in/satyam-kumar-a8b25b1b2)  
- Supervised by **Prof. C. Mallikarjuna**, Dept. of Civil Engineering, IIT Guwahati

*This repository is part of an ongoing M.Tech thesis. Code and notebooks are being cleaned for public release. If you are working on trajectory prediction for Indian/heterogeneous traffic and want to collaborate or discuss, feel free to open an issue or reach out via email.*

---

*Keywords: trajectory prediction · heterogeneous traffic · graph attention networks · Indian urban traffic · safety-aware deep learning · NGSIM · Warangal · autonomous vehicles · ATMS · ITS*
