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
graded explicitly, and GAMUS cannot support it. Measured across every tile we hold, our own
scene classifier labels **87% of GAMUS train, 87% of val and 89% of test as urban**, with
forest-dominated tiles at 19 of 5 004 training tiles (0.4%), none in val and one in test.
Three American cities do not contain forest, hills or open farmland.

Tree *pixels* are abundant — 25% of training pixels — so the model sees plenty of trees; what
it never sees is a forest. Our Veluwe result (2.26 m, r 0.79) is therefore earned almost
entirely by the AHN tiles, not by GAMUS. AHN is the only source we have for three of the four
landscapes the evaluation grades, which makes quadrupling it the most direct way to support a
stability claim.

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

**Semantic priors for water and roads (considered as E4, rejected).** Water is perfectly flat
at a known level and we predict it 0.91 m *above* ground, so clamping it looked like free
accuracy. Two measurements killed it:

* SegFormer (ADE20K) is as domain-mismatched as the COCO detector was. On a nadir harbour scene
  it labels **35% of the image "sky"** and only 22% "sea", against a ground truth of 19% of
  pixels below 0.5 m. Building a prior on that would inject noise, not remove it.
* Even a *perfect* water prior is worth almost nothing. Ranking each class by its contribution
  to total squared error:

| Class | Pixels | RMSE | Bias | Share of squared error |
|---|---|---|---|---|
| Building | 69.2 M | 7.27 m | −2.47 m | **48.8%** |
| Tree | 60.4 M | 6.85 m | −3.98 m | **37.8%** |
| Ground | 82.7 M | 2.24 m | −0.06 m | 5.5% |
| Road | 66.7 M | 2.34 m | −0.23 m | 4.9% |
| Low vegetation | 65.3 M | 1.71 m | +0.36 m | 2.6% |
| Water | 6.9 M | 2.31 m | +0.91 m | **0.5%** |

Buildings and trees together are **87%** of the error. Water is 0.5%, and roads and ground are
already near-unbiased. A rule-based prior for the flat classes optimises the 13% we have
largely solved, at the risk of visible false positives — flattening a dark roof to ground level
looks far worse than being 0.9 m high on a canal. Effort belongs on buildings, which is E1.

**Shadow-length features.** The most principled route to absolute height from one nadir view,
but it is a research project, not an engineering task, days before a deadline.

---

## Where the runs actually happened

The three ablations were meant to run in parallel on a workstation with a GTX 1080, an RTX
3070 and a second GTX 1080. Thirteen minutes into the first run the 3070 fell off the PCIe bus
(`Xid 79`), and the driver marked every card `Node Reboot Required` (`Xid 154`). The two 1080s
still appeared in `nvidia-smi`, but `torch.cuda.is_available()` returned `False` — so both runs
silently fell back to CPU and kept going at roughly seventeen times slower. E1 would have taken
78 hours. This was the third time that card had failed the same way under load.

The experiments moved to three parallel Kaggle sessions on T4s. That turned out better for E2:
a T4 has working mixed precision, so it runs the full four-block, thirty-epoch version that the
1080 could not.

All three runs are otherwise identical — same data sources, same architecture, thirty epochs,
effective batch 36 (12 × 3 accumulation) — so each result is attributable to the single thing
that experiment changes.

| Run | The one change | Everything else |
|---|---|---|
| E1 | `--height-weight 0.5 --height-scale 10` | identical |
| E2 | `--unfreeze-blocks 4 --backbone-lr 3e-5` | identical |
| E3 | AHN 2 000 → 8 000 and 800 → 3 000 tiles | identical |

Each session ends by running `scripts/benchmark_ahn.py` against **its own** checkpoint — the
`--weights` flag added for this, without which every run would have scored the shipped model
and produced three plausible, meaningless results.

## Results

Filled in as runs complete. A run ships only if it meets the criterion stated above.

| Run | Change | AHN 8-site RMSE | mean r | Best val RMSE | Shipped |
|---|---|---|---|---|---|
| baseline | frozen DA2-B + multiscale head | **5.68 m** | 0.662 | 3.850 m | **current** |
| **E1** | **height-weighted loss** | **5.50 m** | 0.656 | **3.809 m** | **candidate** |
| **E2** | **unfrozen last 4 blocks** | **5.68 m** | **0.674** | 3.866 m | **no** |
| E3 | 4× AHN tiles | 5.75 m | 0.663 | 3.871 m* | **no** |

### E2 — unfreezing the backbone: no effect

Thirty epochs with the last four transformer blocks trainable at a tenth of the head's learning
rate, on a Kaggle T4 with mixed precision — the experiment as designed, not the version the
dying 1080 would have given us. 417 s per epoch, so the extra 28 M parameters receiving
gradients cost far less time than expected.

The result is a tie. Mean RMSE over the eight held-out sites is unchanged at 5.68 m, MAE is
0.01 m worse, best validation RMSE is 0.016 m worse. Correlation improves slightly, 0.662 to
0.674, which says the fine-tuned features track the shape of the terrain marginally better
without being any closer in absolute metres.

**It does not ship.** The criterion was to beat the baseline, and a tie is not a win — least of
all for the one experiment that would grow the shipped checkpoint from 6.6 MB to about 390 MB.

**What it tells us, which is the more useful part.** The hypothesis was that the frozen
backbone is the ceiling: trained on eye-level photographs, it correlates only 0.41 with real
height in nadir views, and the head can only work with what it is handed. Adapting those
features to nadir geometry should therefore have raised the ceiling.

It did not. Four blocks and thirty epochs of fine-tuning moved the benchmark by nothing. Either
the frozen features already carry the information the head needs, and the limit lies elsewhere,
or the adaptation needs far more data than 7 804 tiles to take hold. The first reading is the
more useful one and matches the other evidence: what limits us is buildings and trees, where a
single nadir view genuinely lacks the parallax to resolve absolute height — not the backbone's
representation.

That makes E1's premise — that the *loss* under-weights tall structures — the better bet, and
E1 is where the gain actually appeared.

### E1 — height-weighted loss: the one that works

Thirty epochs with each pixel's L1 term weighted by `1 + h/10`, so a 30 m building counts four
times a kerb instead of the roughly equal weight `log1p` was giving them. 404 s per epoch —
the extra term costs nothing measurable.

| | baseline | E1 | change |
|---|---|---|---|
| AHN 8-site mean RMSE | 5.68 m | **5.50 m** | **−0.18 m** |
| AHN 8-site mean MAE | 3.26 m | **3.17 m** | −0.09 m |
| mean correlation | 0.662 | 0.656 | −0.006 |
| best validation RMSE | 3.850 m | **3.809 m** | −0.041 m |

A 3.2% reduction in RMSE and 2.8% in MAE, reproduced on both the validation split and the eight
held-out LiDAR sites, which are different data measured by a different instrument. Two
independent measurements moving together is what distinguishes this from E2's noise-level tie.

The slight correlation dip is the signature of the intended mechanism rather than a
counter-indication. Correlation is scale- and offset-free: it measures whether the predicted
surface has the right *shape*. RMSE and MAE measure whether it is at the right *height*. A
change that improves the latter while very slightly costing the former has moved absolute
heights towards the truth — which is exactly what weighting tall pixels was meant to do, and
exactly what the −2.47 m building bias called for.

**The risk the plan named has not been checked yet.** The concern written down before the run
was that attention moved towards tall objects comes out of ground, road and low vegetation —
sub-metre today, the majority of pixels in any image, and what makes the output *feel*
accurate. The Kaggle cells ran `benchmark_ahn.py` but not `eval.py` on the GAMUS test split, so
the per-class figures the second half of the criterion needs do not exist yet. Until they do,
E1 is a **candidate, not a decision**: `scripts/compare_checkpoint.py` will apply both halves of
the rule mechanically once the checkpoint is in hand.

### E3 — four times the AHN data: slightly worse

8 000 AHN tiles at 0.3 m and 3 000 at 1 m, against the baseline's 2 000 and 800. With GAMUS
unchanged that is 16 004 training tiles per epoch against 7 804, and epoch time rose in exact
proportion — 830 s against 404 s, a factor of 2.05 for 2.05× the data. Nothing was silently
truncated; the run really did see all of it.

| | baseline | E3 | change |
|---|---|---|---|
| AHN 8-site mean RMSE | 5.68 m | 5.75 m | **+0.07 m worse** |
| AHN 8-site mean MAE | 3.26 m | 3.31 m | +0.05 m worse |
| mean correlation | 0.662 | 0.663 | +0.001 |

\* E3's best validation RMSE of 3.871 m is **not comparable** to E1's 3.809 m or E2's 3.866 m.
All three cap validation at 200 tiles, but E3 draws them from a different AHN validation pool
(400 + 150 tiles rather than 200 + 80), so it is a different measurement, not a worse score on
the same one. Only the eight-site LiDAR benchmark is common to all four runs, and that is what
the table above and the ship criterion use.

**It does not ship** — it is worse, not merely tied.

**What it tells us.** The hypothesis was the safest of the three: more of the data that already
worked, sampled across the landscape types the PS grades, to compensate for GAMUS being 87%
urban. Doubling the training set moved the benchmark backwards by 0.07 m.

**First, how much of that is real.** 0.07 m is 1.2% of the baseline, and we have no
seed-repeated run to estimate run-to-run variation, so we cannot distinguish "slightly worse"
from "the same". The claim this result supports is **more data did not help** — the same
conclusion as E2, not a stronger one. A seed repeat is cheap on Kaggle and is the honest way to
put an error bar on all three results; it has not been run.

**Three explanations that do not survive checking.**

*Overfitting.* The natural first guess, and the wrong one. More data is the standard remedy for
overfitting, so quadrupling AHN and getting no gain is the opposite of that mechanism. The
signature is absent too: overfitting shows validation loss rising while training loss falls, and
E3's validation RMSE was still improving at the end (3.894 → 3.884 over the last two epochs).
The trainable head is 6.6 MB against 16 004 tiles — far from a capacity-limited regime — and E2
adding 28 M trainable parameters changed nothing, which is independent evidence that we are not
near an overfitting boundary.

*Under-training.* The learning rate is cosine-annealed per epoch over 30 epochs
(`train_decoder.py:144`), and E3 had 2.05× the optimiser steps within each one. It received
more gradient updates on the same schedule shape, not fewer.

*A shift in the training mixture.* AHN's share went from 36% to 69% of tiles. This was my first
explanation and it is **wrong**: the eight benchmark sites are themselves Dutch AHN/PDOK data,
so a mixture shift towards Dutch imagery should have *helped* on this benchmark, not hurt.
Nor did the AHN distribution itself change — `build_ahn_dataset.py` samples fixed regional
shares (30% urban, 20% hilly, 20% forested, 15% sparse, 15% mixed), so 11 000 tiles are drawn
from the same five regions in the same proportions as 2 800. More tiles, same distribution.

**What is left, and it is the same wall E2 hit.** Four times the tiles, drawn from the same
distribution, carry roughly the same information. The limit is not how many Dutch examples the
head has seen; it is that a single nadir view of a building or a tree canopy does not contain
the parallax that fixes its absolute height, and no quantity of examples of that ambiguity
resolves it.

This matters for the evaluation data specifically. If more Dutch tiles do not help, the gap that
remains is not a shortage of examples — it is a shortage of *variety*, and in particular of the
terrain the PS names that neither GAMUS nor AHN contains: real relief beyond South Limburg, and
Indian urban form. More of either existing source will not close it.

This matters for the evaluation data specifically. If more Dutch tiles do not help, the gap that
remains is not a shortage of examples — it is a shortage of *variety*, and in particular of the
terrain the PS names that neither GAMUS nor AHN contains: hills, and Indian urban form. More of
either existing source will not close it.

---

## Conclusions

| Run | What it changed | 8-site RMSE | Verdict |
|---|---|---|---|
| baseline | — | 5.68 m | |
| **E1** | **the loss function** | **5.50 m** | **the only gain** |
| E2 | model capacity (unfroze 4 blocks) | 5.68 m | no effect |
| E3 | training data volume (2.05×) | 5.75 m | no gain (−1.2%, within unmeasured noise) |

The three experiments changed three different things — the objective, the capacity, the data —
and only the objective mattered. Neither letting the backbone adapt nor doubling the training
set moved the benchmark at all, while re-weighting the loss towards tall structures moved it
0.18 m in one run costing no extra time — two and a half times E3's regression, in the opposite
direction.

That is a coherent finding rather than three isolated results. The model was not short of
capacity or short of examples; it was optimising a score that barely cared about the pixels
carrying most of the error. `log1p` compressed a 60 m building error to roughly three times the
penalty of a 0.5 m kerb error, so the head was behaving correctly for the objective it was
given, and the −2.47 m building bias was the honest consequence.

The practical reading for anything done after this: further gains are more likely to come from
what the model is asked to optimise, or from data of a *kind* it has never seen, than from
more parameters or more of the same tiles.
