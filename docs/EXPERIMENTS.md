# Ablation plan and results

Running record of every training experiment: what was run, why, what it cost, and whether it
shipped. Written before the runs started so the reasoning can be judged independently of the
outcome.

---

## What we are optimising for

The problem statement (PS 26175) splits marks evenly:

* **DSM estimation accuracy — 50%.** "Evaluate RMSE, MAE, and correlation against LiDAR or
  reference data, **including performance stability across urban, sparse, hilly, and forested
  landscapes**."
* **Rendering and user experience — 50%.** Projection accuracy, visual fidelity, flythrough
  navigability, interface, software stability, standalone deployment.

And, decisively: **"During final evaluation, ISRO RGB-band optical satellite imagery will be
used."** We do not choose the test data. Whatever we train must survive imagery we have never
seen, at satellite resolution, over Indian terrain.

Two consequences shape every decision below:

1. **An experiment only counts if it improves the eight-site LiDAR benchmark**, which is
   held out, in a different country, from a different sensor. Improving a validation split we
   also tuned against would be self-deception.
2. **Anything that slows inference or bloats deployment costs us marks on the other 50%.** A
   model that is 0.1 m better but four times slower is a bad trade for this submission.

### Where the error actually is

From the current shipped checkpoint (§4.4, §4.5 of the report):

| Surface | RMSE | Bias | Share of the problem |
|---|---|---|---|
| Ground / road / low vegetation | 1.7–2.3 m | ≈ 0 | already sub-metre in MAE; little left to win |
| **Building** | **7.27 m** | **−2.47 m** | dominant, and systematically low |
| **Tree** | **6.85 m** | **−3.98 m** | dominant, and systematically low |

By landscape, the weak case is dense urban (8.52 m) and high-rise (16.94 m); forest (2.26 m),
hills (3.41 m) and sparse (1.53 m) are already good.

**So the target is buildings.** Not 100 m towers — Indian urban fabric is mostly 3–8 storeys,
roughly 10–25 m — but the systematic −2.5 m under-call on ordinary buildings, which will appear
in nearly every urban tile ISRO supplies.

---

## Baseline

`checkpoints/decoder/decoder_best.pth` — frozen Depth Anything V2 Base + 1.72 M-parameter
multi-scale head, 7 804 training tiles, 30 epochs.

| Metric | Value |
|---|---|
| AHN benchmark, 8 sites, 0.25 m | **5.68 m RMSE**, 3.26 m MAE, r 0.662 (predict-zero 8.37 m) |
| Same at 1 m imagery | 5.94 m RMSE, r 0.634 |
| GAMUS test, 500 tiles | 5.42 m RMSE, 2.46 m MAE, r 0.751 |
| Inference | ~1.3 s per 512² tile (MPS); head is 6.6 MB |

**Ship criterion for every experiment below: beat 5.68 m mean RMSE on the eight-site benchmark
without losing more than 0.05 m on the sub-metre classes (ground, road, low vegetation) and
without a material increase in inference cost.** Otherwise the baseline ships.

---

## Experiments

### E1 — Height-weighted loss

**Hypothesis.** The dominant loss term is SILog computed on `log1p` heights. Log compresses
large values, so a 60 m error on a 100 m building registers as only ~3× the penalty of a 0.5 m
error on a 1 m kerb, although it is 120× larger in metres. The model is therefore correctly
optimising a score that barely cares about buildings, and the −2.47 m building bias is the
predictable result. Weighting each pixel's loss by its true height should recover some of it.

**Why it is relevant.** Buildings are the largest error term, and "urban" is the first landscape
the PS names. This is the most directly targeted change available.

**Expected gain.** Building RMSE down; overall bias closer to zero.

**Cons and risks.**
* Ground, road and low vegetation are currently sub-metre — the best part of the model, and the
  majority of pixels in any image. Shifting attention towards tall objects can degrade them.
* Median error (0.64 m today) may rise even if RMSE falls. Median is what makes the product
  *feel* accurate.
* Over-weighting could cause over-prediction on flat roofs, which would look worse in the 3D
  view than under-prediction does.

**Feasibility.** High — one file (`heimdall/decoder/loss.py`), one training run, ~5 h.

---

### E2 — Unfreeze the last transformer blocks

**Hypothesis.** The backbone is frozen and was trained on natural, eye-level photography. Nadir
aerial imagery is a large domain shift: no horizon, no perspective, objects seen only from
above. Measured correlation of the frozen backbone with real height is 0.41. That is the
ceiling on what any head can extract. Fine-tuning the last few blocks at a low learning rate
should let the features themselves adapt to nadir geometry.

**Why it is relevant.** It is the only experiment that raises the ceiling rather than dividing
the existing signal differently. Highest potential gain of the three.

**Expected gain.** The largest of the three if it works — possibly several tenths of a metre.

**Cons and risks.**
* **Catastrophic forgetting.** The general depth prior is what makes single-view geometry work
  at all; 7 804 tiles is small relative to a 97 M-parameter backbone.
* **Overfitting to US and Dutch imagery**, which is exactly the wrong failure for Indian
  evaluation data. The held-out benchmark is the guard, but it is also Dutch.
* **Deployment cost.** We currently ship a 6.6 MB head. Fine-tuned blocks mean shipping backbone
  weights too — hundreds of megabytes in the Docker image and the repo. That is a real cost
  against the "standalone deployment" criterion.
* **VRAM.** Backbone gradients on an 8 GB card force small batches.

**Feasibility.** Medium. Needs the 3070 (Ampere, mixed precision actually helps). ~8 h.

---

### E3 — More AHN data

**Hypothesis.** We trained on 2 800 Dutch tiles. PDOK is open and effectively unlimited, so the
cheapest available lever is simply more of the data that already worked, sampled across the
landscape types the PS grades.

**Why it is relevant.** "Stability across urban, sparse, hilly and forested landscapes" is
graded explicitly, and GAMUS is three US cities with exactly one forested tile in its test
split. Broader sampling is the honest way to support a stability claim.

**Expected gain.** Modest but reliable; most likely to show up as landscape consistency rather
than headline RMSE.

**Cons and risks.**
* **Dutch bias.** The Netherlands is flat, architecturally uniform and has no dense mid-rise of
  the Indian kind. Over-representing it could make the model *worse* on ISRO's imagery while
  looking better on our Dutch benchmark — the one failure mode our evaluation cannot detect.
  Mitigation: cap AHN at roughly half the training mix and keep GAMUS dominant.
* Download time (hours), and diminishing returns are likely.

**Feasibility.** High — no new code, only `scripts/build_ahn_dataset.py` with larger counts.

---

### Considered and rejected

**DA2-Large backbone.** Equal measured frozen correlation (0.41 vs 0.41) but ~4× slower
inference (6.6 s vs 1.6 s per tile). Job latency is part of the graded user experience, and the
evidence for an accuracy gain is absent. Not worth the trade.

**Test-time augmentation.** Already measured: 0.02 m *worse* for 2.3× the runtime. Rejected on
evidence, documented in the report.

**Shadow-length features.** The most principled route to absolute height from one nadir view,
but it is a research project, not an engineering task, days before a deadline.

---

## Results

Filled in as runs complete. A run ships only if it meets the criterion stated above.

| Run | Change | AHN 8-site RMSE | r | Sub-metre classes | Inference cost | Shipped |
|---|---|---|---|---|---|---|
| baseline | frozen DA2-B + multiscale head | **5.68 m** | 0.662 | ground 0.80 / road 0.86 / low-veg 0.75 m MAE | 1.3 s/tile, 6.6 MB | current |
| E1 | height-weighted loss | pending | | | | |
| E2 | unfrozen last blocks | pending | | | | |
| E3 | +10 k AHN tiles | pending | | | | |
