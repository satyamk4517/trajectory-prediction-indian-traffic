# Reference Papers

The PDFs of these papers live in the project's private knowledge base. This file is a self-contained reference list — every line below is a model or idea that directly influenced the architecture in this repository.

---

## Foundational social-aware models

**Social-LSTM** — Alahi et al., *Social LSTM: Human trajectory prediction in crowded spaces*, CVPR 2016.
First widely-cited social-pooling architecture. Establishes the per-agent LSTM + neighbourhood pooling pattern.

**Social GAN** — Gupta et al., *Social GAN: Socially acceptable trajectories with generative adversarial networks*, CVPR 2018.
Generative multi-modal extension of Social-LSTM.

**Convolutional Social Pooling (CS-LSTM)** — Deo & Trivedi, *Convolutional social pooling for vehicle trajectory prediction*, CVPR-W 2018.
The de facto NGSIM baseline for years. CS-LSTM @5s ADE ≈ 1.95 m on NGSIM.

**Social Attention / SoPhie** — Sadeghian et al., *SoPhie: An attentive GAN for predicting paths compliant with social and physical constraints*, CVPR 2019.
Attention-based replacement for max/mean pooling.

---

## Heterogeneous / class-aware modelling

**TrafficPredict** — Ma et al., *TrafficPredict: Trajectory prediction for heterogeneous traffic-agents*, AAAI 2019.
Inspired the heterogeneous neighbour-class encoding used in SP-GAT (Phase 18+).

**SR-LSTM** — Zhang et al., *SR-LSTM: State refinement for LSTM towards pedestrian trajectory prediction*, CVPR 2019.
State-refinement idea — informs the iterative-refinement decoder.

---

## Multi-modal / target-conditioned models

**DESIRE** — Lee et al., *DESIRE: Distant future prediction in dynamic scenes with interacting agents*, CVPR 2017.
Multi-modal CVAE backbone. Informed the Winner-Takes-All (WTA) 3-mode loss in NGSIM K-PhysGAT.

**Trajectron / Trajectron++** — Salzmann et al., 2020.
Dynamically-feasible CVAE with integrated bicycle dynamics. The single closest spiritual ancestor of K-PhysGAT.

**TNT (Target-driveN Trajectory)** — Zhao et al., ECCV 2020.
Goal-anchored multi-modal prediction.

**Multi-Agent Tensor Fusion (MATF)** — Zhao et al., CVPR 2019.
Tensor-based multi-agent representation.

---

## Safety-aware losses

**RAPiD** — *Reliable, Aware, Physically-Informed Decision-making for trajectory prediction.*
Direct inspiration for the Phase 22 safety loss suite (ACT + proximity + jerk). The combination of geometric accuracy with anticipated-collision-time, proximity, and smoothness penalties is the RAPiD-style multi-objective view, adapted here to per-class weights.

---

## How each paper influenced this work

| Component | Paper(s) |
|---|---|
| BiLSTM motion encoder | Social-LSTM, CS-LSTM |
| Edge-feature multi-head GAT | (new in this work; closest analogue: SoPhie attention) |
| Heterogeneous neighbour class embedding | TrafficPredict |
| Differentiable kinematic bicycle rollout | Trajectron++ |
| Multi-mode WTA decoder (K=6) | DESIRE, TNT |
| ACT + proximity + jerk loss | RAPiD |
| Horizon-weighted Huber | (engineering choice; common in trajectory work) |

---

## BibTeX

```bibtex
@inproceedings{alahi2016social,
  title={Social LSTM: Human trajectory prediction in crowded spaces},
  author={Alahi, Alexandre and Goel, Kratarth and Ramanathan, Vignesh and
          Robicquet, Alexandre and Fei-Fei, Li and Savarese, Silvio},
  booktitle={CVPR},
  year={2016}
}

@inproceedings{deo2018cslstm,
  title={Convolutional social pooling for vehicle trajectory prediction},
  author={Deo, Nachiket and Trivedi, Mohan M.},
  booktitle={CVPR Workshops},
  year={2018}
}

@inproceedings{ma2019trafficpredict,
  title={TrafficPredict: Trajectory prediction for heterogeneous traffic-agents},
  author={Ma, Yuexin and Zhu, Xinge and Zhang, Sibo and Yang, Ruigang and Wang, Wenping and Manocha, Dinesh},
  booktitle={AAAI},
  year={2019}
}

@inproceedings{salzmann2020trajectron,
  title={Trajectron++: Dynamically-feasible trajectory forecasting with heterogeneous data},
  author={Salzmann, Tim and Ivanovic, Boris and Chakravarty, Punarjay and Pavone, Marco},
  booktitle={ECCV},
  year={2020}
}

@inproceedings{gupta2018socialgan,
  title={Social GAN: Socially acceptable trajectories with generative adversarial networks},
  author={Gupta, Agrim and Johnson, Justin and Fei-Fei, Li and Savarese, Silvio and Alahi, Alexandre},
  booktitle={CVPR},
  year={2018}
}

@inproceedings{lee2017desire,
  title={DESIRE: Distant future prediction in dynamic scenes with interacting agents},
  author={Lee, Namhoon and Choi, Wongun and Vernaza, Paul and Choy, Christopher B. and Torr, Philip H. S. and Chandraker, Manmohan},
  booktitle={CVPR},
  year={2017}
}
```
