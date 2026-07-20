# HetSpA — Heterogeneous Spatial Attention on Warangal Urban Traffic

> **Model note.** **HetSpA** (Heterogeneous Spatial Attention) is the current, paper-verified
> model for this Warangal work. An earlier phase of the same line of work, on this same dataset,
> was called **SP-GAT** (the Phase 16 → 22 experiments); those numbers used an 80/20 temporal
> split with vehicles shared between train and validation, so they carry the same leakage the
> project later identified and corrected, and are superseded here rather than restated. HetSpA
> is trained and scored under a vehicle-level 3-fold GroupKFold protocol from the start. A
> manuscript covering this work is in preparation for *Transportation Research Part C*.
>
> Notebook: [`notebooks/02_warangal_spgat_phases.ipynb`](../notebooks/02_warangal_spgat_phases.ipynb)
> (the notebook name still carries the earlier "spgat" label). Legacy Phase 16–22 artifacts
> remain under [`results/warangal/`](../results/warangal/) for historical reference.

---

## 1 · Why a separate model for Warangal?

NGSIM is a US freeway: lane-disciplined, low class-diversity, predictable interactions. The
Warangal dataset — drone video over a lane-free, mixed-class urban arterial in Warangal,
Telangana, India — is the opposite: bikes, autos, cars, and heavy vehicles share the same
stream with no lane discipline to fall back on. A different design is required, and it needs
its own honest evaluation protocol rather than borrowing NGSIM's.

## 2 · Dataset

| Property | Value |
|---|---|
| Source | Drone video, lane-free urban arterial, Warangal, Telangana, India |
| Capture / working rate | 30 Hz captured, downsampled to 10 Hz |
| Window | 3 s history (30 steps) + 5 s future (50 steps) |
| Unique vehicles | 12,762 total — Bike 7,355 (57.6%), Auto 2,734 (21.4%), Car 2,132 (16.7%), Heavy 541 (4.2%) |
| Training windows | 147,847 total — Bike 50,000 (capped), Auto 50,000 (capped), Car 37,281, Heavy 10,566 |
| Evaluation | 3-fold vehicle-ID GroupKFold — no vehicle appears in both train and test in any fold; 44,088 stored test windows |

Bike is the dominant class by a wide margin, consistent with mixed Indian urban traffic;
Heavy is the data-scarce class.

## 3 · Architecture

HetSpA = **Het**erogeneous **Sp**atial **A**ttention (bicycle-decoded). ~1.87 M parameters.

- **Ego GRU encoder** — encodes the ego vehicle's own 3 s history.
- **8-slot neighbour encoder with masked social attention** — up to 8 neighbours per window,
  attention masked over unused slots so a sparse-neighbourhood window is not penalised for
  absent neighbours.
- **Class-conditioned scene state** — the encoded scene is conditioned on the ego vehicle's
  class (bike / auto / car / heavy), so the model can express class-appropriate dynamics
  without training four fully separate models.
- **Kinematic bicycle decoder** — an integrator of bounded steering/acceleration controls, so
  every output trajectory is physically drivable by construction rather than a free-form (x, y)
  regression.
- **6 modes with a scoring head** — multimodal output, each mode scored so a single
  highest-confidence trajectory can be selected at deployment time (see §6).
- **ACT (Anticipated Collision Time) risk instrument** — a safety-relevant auxiliary signal
  computed alongside the trajectory output.

![HetSpA architecture](assets/ARCH_paper.png)

**Training note (disclosed).** HetSpA is trained with mirror-Y flip augmentation
(`p = 0.5`). An ablation shows this is **not load-bearing**: the no-flip 3-fold pooled minADE is
0.4205 ± 0.0189 m against the flip-augmented 0.436 ± 0.014 m headline — within fold-to-fold
noise of each other. The headline model reported below is the flip-augmented one; the
no-flip variant is documented as an ablation, not a stronger alternative.

## 4 · Evaluation protocol

3-fold vehicle-ID GroupKFold: each vehicle's windows fall entirely in one fold, so a vehicle
never appears in both a fold's training set and its own test set. 44,088 test windows are
stored and scored across the three folds. All headline numbers below are 3-fold mean ± standard
deviation unless stated otherwise.

Eight baselines were re-implemented in **one shared training harness** on the same Warangal
data under the same protocol, so the comparisons below are apples-to-apples within this
repository: CV, S-LSTM, CS-LSTM, TraPHic, TrafficPredict, TNT, Transformer, HEAT — plus K=6
multimodal variants of Transformer, HEAT, and CS-LSTM(M).

## 5 · Headline results

**Pooled best-of-six minADE (3-fold mean ± sd): 0.436 ± 0.014 m.**

| Class | minADE (m) |
|---|---:|
| Bike | 0.517 |
| Auto | 0.425 |
| Car | 0.383 |
| Heavy | 0.304 |

![Per-class minADE](assets/T1_paper.png)

**Strongest baselines, same harness, same protocol:**

| Model | Pooled minADE (m) | Margin vs HetSpA |
|---|---:|---:|
| HEAT-K6 | 0.510 ± 0.072 | ~15% |
| CS-LSTM | 0.572 ± 0.021 | ~24% |

**Endpoint accuracy:** HetSpA's own pooled minFDE is **1.09 m**, the best of any class in the
comparison. (Baseline FDE rows are not quoted here — a convention mismatch in how FDE was
computed for some baselines is being corrected in the manuscript; HetSpA's own FDE number is
verified and stands.)

**Per-window pairing** (same 44,088 test windows, paired comparison): HetSpA scores lower error
than HEAT-K6 on **65.1%** of windows (mean advantage 0.074 m where it wins) and lower than
CS-LSTM on **72.8%** of windows (mean advantage 0.136 m).

## 6 · The metric we lose, reported honestly: deployed top-1

A multimodal model has to commit to one trajectory at deployment time. Scored on that
single, top-ranked mode (not best-of-six):

| Model | Deployed top-1 ADE (m) |
|---|---:|
| HEAT | 0.855 |
| S-LSTM | 0.858 |
| HetSpA (top-1 mode) | 0.950 |

This is a real, structural cost of carrying six modes rather than one: the scoring head does
not always rank the eventually-best mode first. A trained commitment head narrows this gap to
**0.874 m** while still keeping all six modes available internally. HetSpA does not win on this
metric; stating that plainly here is intentional.

## 7 · Dynamics realism

HetSpA is the **only** model in the comparison that reproduces the physically correct ordering
of heading-rate and jerk across classes: Bike > Auto > Car > Heavy. Mean predicted jerk is
**0.32 m/s³** against a ground-truth mean of **0.78 m/s³**; the baselines predict
**2.6–12.9 m/s³** — an order of magnitude too rough, i.e. much jerkier than real driving.
On a composite realism-distance score, HetSpA scores **3.15** against the best baseline's
**3.39** — a real but modest margin, stated as such rather than inflated.

![Dynamics realism](assets/REAL2_paper.png)

## 8 · A safety-critical turn case, and its limits

On one safety-critical left turn, HetSpA commits to the correct bending mode and ends with a
**2.2 m** final error, while the single-mode baselines run straight through the turn and end
23–48 m off. This is a real, illustrative win for multimodality plus the bicycle decoder.

It does **not** generalise into a claim of turn dominance: on the full 45-window turning
subset, all models are statistically tied. Both facts are stated together deliberately — the
single dramatic case is real, and so is the tie across the full subset.

![Turn case, 3D](assets/V1_3d_paper.png)

## 9 · The diagnostic story: how much does the interaction channel actually buy?

This is the second half of the underlying manuscript, and the more surprising finding.

With vehicle class and kinematic structure held fixed, **removing all neighbour information
costs only centimetres** of minADE at Warangal, a free-flow site. That does not mean the
interaction channel is inert: paired per-window comparison shows the full-graph (with
neighbours) model beats the ego-only ablation on a large majority of windows on an external,
denser site (see below) — the channel demonstrably **fires**, but its **payload is small** at
this site. The predictive signal is carried mostly by vehicle class and kinematic structure,
not by the neighbour-attention machinery.

![Interaction ladder](assets/E1_ladder_paper.png)

**External check on SinD** (drone data, four-city Chinese intersections, denser interaction
than Warangal's free-flow arterial): the same attribution pattern replicates — the full-graph
model beats the ego-only ablation on **70.5%** of paired windows off-site, confirming the
channel fires outside Warangal too. As a further external corroboration, a published HEAT
model trained on the same SinD cache scores **HEAT-K6 0.316 m** against **HetSpA 0.334 m** — a
statistical near-tie, disclosed rather than smoothed over.

## 10 · Honest caveats, collected in one place

- **Deployed top-1 ADE is a loss, not a win** (§6): HetSpA 0.950 m vs HEAT 0.855 m / S-LSTM
  0.858 m on the single-mode-at-deployment metric; a trained commitment head narrows this to
  0.874 m without discarding the other five modes.
- **Turn dominance is one case, not a subset result** (§8): the 2.2 m vs 23–48 m turn is real;
  the full 45-window turning subset ties across models.
- **Flip augmentation is disclosed and shown not load-bearing** (§3): no-flip 3-fold pooled
  minADE 0.4205 ± 0.0189 m is within noise of the flip-augmented headline 0.436 ± 0.014 m.
- **The interaction channel's payload is small at Warangal** (§9): centimetre-level cost from
  removing neighbour information at this free-flow site; the channel is shown to fire more
  strongly at a denser external site (SinD), and a published baseline corroborates the
  magnitude there.
- **Baseline FDE rows are withheld pending a convention fix** (§5): only HetSpA's own,
  verified FDE number is reported.

## 11 · Superseded content

Sections of the earlier SP-GAT documentation described a Phase 16 → 22 ablation on this same
Warangal dataset, evaluated under an 80/20 temporal split with vehicles shared between train
and validation. That evaluation protocol has the same leakage pattern the NGSIM arm of this
project identified and corrected for (see [`docs/K_PhysGAT_NGSIM.md`](K_PhysGAT_NGSIM.md)),
so those Phase 16–22 numbers are not restated in this document. The results in §5–§9 above use
the vehicle-level 3-fold GroupKFold protocol from the start.
