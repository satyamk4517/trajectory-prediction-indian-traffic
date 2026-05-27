# SP-GAT — Social-Physics GAT on Warangal Heterogeneous Urban Traffic

> Methodology and evaluation notes for the Warangal arm of the project.
> Notebook: [`notebooks/02_warangal_spgat_phases.ipynb`](../notebooks/02_warangal_spgat_phases.ipynb)
> Results: [`results/warangal/`](../results/warangal/)

---

## 1 · Why a separate model for Warangal?

NGSIM is a US freeway: lane-disciplined, low class-diversity, predictable interactions. The Warangal dataset (drone video over a 2-lane, 2-way undivided urban arterial in Warangal, Telangana, India) is the opposite — bikes, autos, cars, and buses weaving in a lane-less stream where a 200 kg motorbike shares a 4 m lane with a 12 t bus. Direct deployment of NGSIM-trained models gives ADE > 8 m on this data. A different design is required.

## 2 · Architecture changes vs the NGSIM K-PhysGAT backbone

Five class-specific additions:

1. **Per-class training** — separate model weights for bike / auto / car / bus. A pooled model was tried and lost on every class. Bus dynamics (slow, heavy, deterministic) and bike dynamics (fast, light, weaving) are too different to share parameters.
2. **Heterogeneous neighbour-class encoding** — every neighbour edge carries a one-hot class indicator, passed through a learned 16-dim embedding (TrafficPredict-inspired). A "bike near a bus" is treated structurally differently from "bike near another bike".
3. **Physics-informed initialisation** — `MAX_V0 = 40 m/s` (covers real Car/Bus speeds without clipping), and RandomUniform init on the control Dense layer to escape the zero-gradient deadlock that an all-zero init under `tanh` saturation creates.
4. **Mirror-Y data augmentation** — for a symmetric 2-way road, reflecting trajectories across the road's Y-axis doubles the effective dataset with no new data collection. **Highest-ROI change in the entire ablation** for the data-scarce Bus class (only 13 unique bus vehicles in the source video).
5. **Multi-objective safety loss** (Phase 22) — adds ACT (Anticipated Collision Time), proximity, and jerk penalties to the geometric loss. Weights are **class-aware**: bikes tolerate higher jerk than buses, and the loss should reflect that.

## 3 · Phase-by-phase ablation (P16 → P22)

Starting from an early multi-agent social LSTM baseline averaging ~8 m ADE across classes:

| Phase | Key change | Bike ↓ | Auto ↓ | Car ↓ | Bus ↓ |
|---|---|---:|---:|---:|---:|
| Baseline | Multi-agent social LSTM | 8.83 | 8.72 | 7.85 | 6.96 |
| P16 | **Composite string IDs** (was the dominant bug) | 2.247 | 2.358 | 3.706 | 6.765 |
| P17 | + physics init + macro heading + per-class training | 1.111 | 0.877 | 1.330 | 1.298 |
| P18 | + heterogeneous neighbour one-hot + 16-dim embedding | 1.369 | 1.393 | 1.074 | 1.348 |
| P19 | + mirror-Y augmentation (2× data) | 1.113 | **0.773** ⭐ | 0.911 | 0.606 |
| P21 | + ACT safety penalty | 1.045 | 0.812 | **0.805** ⭐ | 0.522 |
| **P22** | **+ proximity + jerk (RAPiD-style safety suite)** | **1.029** ⭐ | 0.814 | 0.906 | **0.496** ⭐ |

Raw CSV: [`results/warangal/ablation_phases_16_to_22.csv`](../results/warangal/ablation_phases_16_to_22.csv).

### Best-ever per class

| Class | Best phase | minADE (m) | minFDE (m) | RMSE (m) | vs baseline |
|---|---|---:|---:|---:|---:|
| 🚌 Bus | P22 | **0.496** | 0.960 | 0.686 | **−92.9%** |
| 🛺 Auto | P19 | **0.773** | 1.866 | 1.167 | **−91.1%** |
| 🚗 Car | P21 | **0.805** | 2.097 | 1.235 | **−89.7%** |
| 🚲 Bike | P22 | **1.029** | 2.414 | 1.861 | **−88.3%** |

Raw CSV: [`results/warangal/best_per_class.csv`](../results/warangal/best_per_class.csv).

## 4 · Engineering lessons that drove the ablation

These were not predicted in advance. They were earned through training-log inspection.

#### (a) The composite-ID bug — P16 single biggest win
Numeric-only vehicle IDs collided across classes: `Bike_id = 1` and `Car_id = 1` overwrote each other in the neighbour dictionary, silently corrupting **all** training data. Errors of 23–35 m were not the model's fault. Switching to composite string IDs (`"Bike_1"`, `"Car_1"`) was a one-line preprocessing change that collapsed error by ~10×.
**Lesson:** When the model "doesn't work", suspect the data pipeline before the architecture.

#### (b) Per-class training beat pooled training (P17)
Naive intuition: more data is better, train one model on everything. **Wrong.** Bus and bike live in different speed and dynamics regimes; the pooled model under-fit both.
**Lesson:** Distribution heterogeneity can defeat the "more data wins" prior.

#### (c) Mirror-Y was the highest-ROI augmentation (P19)
For a symmetric 2-way road, a Y-axis reflection is a label-preserving transformation. With only 13 unique buses in the data, this doubled effective Bus samples for zero collection cost.
**Lesson:** Exploit dataset symmetries before reaching for synthetic data.

#### (d) Safety losses improved geometric accuracy (P21–P22)
Counter-intuitive: adding ACT + proximity + jerk penalties **lowered ADE/FDE** on Bus and Car. Physical-plausibility regularisation acts as a strong prior matching real driving — improving generalisation, not just safety.
**Lesson:** Domain priors regularise; don't treat "accuracy" and "physical plausibility" as a trade-off.

#### (e) The dummy-`val_loss` silent failure
Keras' compiled-loss reduction returned `val_loss = 0.0` during validation when the custom loss had auxiliary terms. Early stopping was thus driven by a meaningless signal, letting the model overfit invisibly. Fixed by overriding `test_step` explicitly.
**Lesson:** **Always verify `val_loss > 0`** before trusting any training curve.

#### (f) Stride matters
Reducing window stride from 5 to 3 looks like free data but produces highly correlated overlapping samples. The model "memorises" the overlap rather than generalising. **Stride 5 generalised better than stride 3 despite ~40% fewer samples.**
**Lesson:** Sample independence > sample count.

## 5 · Honest evaluation caveat (in-progress fix)

The Phase 16–22 numbers above use an **80/20 temporal sequence split** on the windowed sample array, **not** vehicle-level GroupKFold. The same vehicles appear in train and validation. This is the identical leakage pattern that inflated the NGSIM numbers before the v10 → v14 fix (see [`docs/K_PhysGAT_NGSIM.md`](K_PhysGAT_NGSIM.md)).

**The Phase 17–22 sub-1m ADEs are very likely optimistic.** Phase 23 is migrating Warangal to a vehicle-level GroupKFold protocol — matching the NGSIM v14 evaluation. Expected outcome: numbers degrade somewhat under honest evaluation but become **defensible and publishable**.

Flagging your own evaluation flaws is not optional. Research credibility = protocol integrity.

## 6 · Dataset notes

| Property | Value |
|---|---|
| Source | Drone video, urban arterial, Warangal, Telangana, India |
| Sampling | 25–30 Hz (downsampled to 10 Hz for NGSIM-compatibility) |
| Road | 2-lane, 2-way undivided urban |
| Unique vehicles | 371 Bike · 116 Auto · 126 Car · **13 Bus** (data-scarce) |
| Window: history × future | 30 × 50 frames (3 s × 5 s @ 10 Hz) |
| Stride | 5 frames |
| Neighbours per ego | top-5 within 25 m |
| Samples per class | up to 3 000 (augmented to 6 000 via mirror-Y in P19+) |

For the Bus class with only 13 source vehicles, GroupKFold k=5 leaves 2–3 vehicles per validation fold. Phase 23 will likely use k=3 or leave-one-vehicle-out for Bus specifically.
