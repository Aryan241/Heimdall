# Training the decoder head

## Why retraining is required

The shipped head was trained on a *Depth Anything V3 **Metric*** backbone. Measured against
airborne LiDAR (`scripts/benchmark_ahn.py`, 8 sites, none used for training), that backbone
is nearly blind to height in straight-down imagery:

| Backbone | Type | Mean r | Median r | s/tile (MPS) |
|---|---|---|---|---|
| DA3 Metric-Large (old default) | metric | 0.06 | — | 26 |
| DA3-Large | general / multi-view | 0.23 | 0.29 | 6.1 |
| DA3 Mono-Large | relative | 0.40 | **0.58** | 4.6 |
| **DA2 Base (new default)** | relative | **0.41** | 0.56 | **1.6** |

This is a *relative vs metric* split, not V2 vs V3. Metric models regress absolute distance,
which requires a pinhole camera with a known focal length; an orthorectified mosaic has no
single camera and no perspective, so the metric branch collapses to a near-constant plane
(measured range 0.6–0.7 m across an entire city tile). With no height signal, the head could
only guess from colour and texture: **7.44 m** nDSM RMSE versus **8.37 m** for predicting zero.

DA2 Base and DA3 Mono tie on accuracy; DA2 Base is ~3× faster, so it is the default. To train
against DA3 Mono instead, pass `--model da3-mono-l` everywhere below (a head is fitted to one
backbone's output, so the two are not interchangeable afterwards).

The head cannot be fine-tuned from the old checkpoint — train from scratch.

---

## Datasets

**GAMUS is the primary training set.** The problem statement names it explicitly ("Dataset
Name: GAMUS, Provider: Earthflow"), it is 0.33 m nadir aerial imagery with LiDAR-derived
nDSM — the exact task — and it ships semantic labels used for per-class accuracy reporting.

| Split | Tiles | Use |
|---|---|---|
| train | 5 004 | training |
| val | 859 | epoch validation / checkpoint selection |
| test | 2 861 | final accuracy (`eval.py`) — never train on this |

```bash
python scripts/fetch_gamus.py --out data/GAMUS --splits train val test
```

The HDF5 originals are ~37 GB for train alone. The fetcher streams each tile, converts it to
JPEG + GeoTIFF (+ PNG labels), deletes the HDF5 and resumes where it left off, so the whole
dataset lands in ~9 GB. Use `--tiles N` for a subset on constrained machines.

**AHN is a supplement, not a replacement.** The problem statement allows "any open-source
dataset containing remote-sensing depth data", and GAMUS is US-only. `scripts/build_ahn_dataset.py`
pulls open Dutch aerial photos + AHN LiDAR: a second geography, denser urban fabric, and the
landscape types the evaluation calls out (urban, sparse, hilly, forested). It excludes a 3 km
buffer around every benchmark site so `scripts/benchmark_ahn.py` stays a held-out test.

---

## Path A — multi-GPU workstation (preferred)

One script does everything; each stage is resumable.

```bash
tmux new -s heimdall                         # survives an SSH drop
./scripts/train_workstation.sh 2>&1 | tee outputs/train_run.log
```

Defaults assume `cuda:0` GTX 1080, `cuda:1` RTX 3070, `cuda:2` GTX 1080 — the 3070 trains
(Ampere, so mixed precision actually helps; Pascal cards gain nothing from fp16 and AMP is
skipped automatically), while the two 1080s cache backbone features for the two datasets in
parallel. Override anything:

```bash
TRAIN_GPU=1 CACHE_GPU_A=0 CACHE_GPU_B=2 MODEL=vit-b EPOCHS=40 BATCH=4 \
STAGES=cache,train,bench ./scripts/train_workstation.sh
```

Rough timings on that hardware: datasets ~1 h (network-bound), caching ~20 min, training
~15–25 min per 10 epochs, benchmarks ~20 min.

**Batch size**: see the measured table below — on an 8 GB card use
`--patch-size 256 --batch-size 4 --accumulate 8`. Training also survives a transient OOM
caused by another process on a shared GPU: it drops the batch, waits and continues.

**Worth doing with three GPUs**: train two backbones at once and keep the better one —
`MODEL=vit-b TRAIN_GPU=1 TAG=v2b` alongside `MODEL=da3-mono-l TRAIN_GPU=0 TAG=mono`, then
compare their `benchmark_ahn` tables. They tie in the frozen-backbone test, but which one
trains better on GAMUS is an open question worth settling with evidence.

## Path B — Kaggle (the reference run)

This is the run that produced the shipped checkpoint, so it is known-good rather than
illustrative. Notebook settings: **Accelerator GPU T4 ×2, Internet on**. No Kaggle dataset
needs to be attached — GAMUS comes from Hugging Face and AHN from PDOK. Paste as one
`%%bash` cell, then *Save Version → Save & Run All*; it takes ~4.7 h and fits the 12 h limit.

```bash
%%bash
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /kaggle/working
[ -d heimdall/.git ] || git clone -q https://github.com/Aryan241/Heimdall.git heimdall
cd heimdall || exit 1
pip install -q rasterio scikit-image trimesh
mkdir -p /kaggle/temp/data /kaggle/working/checkpoints

python scripts/fetch_gamus.py --out /kaggle/temp/data/GAMUS --splits train val --workers 6
python scripts/fetch_gamus.py --out /kaggle/temp/data/GAMUS --splits test --tiles 500 --workers 6
python scripts/build_ahn_dataset.py --out /kaggle/temp/data/AHN     --train 2000 --val 200 --workers 12
python scripts/build_ahn_dataset.py --out /kaggle/temp/data/AHN_1m  --train 800  --val 80 --gsd 1.0 \
       --workers 12 --seed 11

python scripts/train_decoder.py \
    --dataset-dir /kaggle/temp/data/GAMUS /kaggle/temp/data/AHN /kaggle/temp/data/AHN_1m \
    --model vit-b --head multiscale --gpu 0 \
    --epochs 30 --batch-size 12 --patch-size 256 --accumulate 3 \
    --lr 3e-4 --num-workers 2 --val-max 200 \
    --output-dir /kaggle/working/checkpoints --tag kaggle

python eval.py --weights /kaggle/working/checkpoints/decoder_best_kaggle.pth \
    --data-dir /kaggle/temp/data/GAMUS --split test --device cuda:0 \
    --output /kaggle/working/gamus_test.json --save-examples 8
```

Notes that matter on Kaggle specifically:

* Datasets go in `/kaggle/temp` (not `/kaggle/working`, capped at 20 GB and persisted into the
  saved version — you do not want 12 GB of tiles in the output).
* Backbone caching is skipped deliberately. It trades disk for time, and on a T4 the forward
  pass through a frozen ViT-B is cheap enough (~7 min/epoch for 9 k tiles) that the cache
  would cost more in download and disk than it saves.
* Add `curl` calls to an [ntfy](https://ntfy.sh) topic around each stage if you want to be
  told when it finishes or fails rather than watching the tab.

### What the reference run produced

30 epochs over 9 004 tiles (GAMUS train + val, AHN 0.33 m, AHN 1 m), ~7 min/epoch:

| | val RMSE | MAE | r |
|---|---|---|---|
| Epoch 1 | 5.26 m | 2.52 m | 0.58 |
| Epoch 25 (**selected**) | **3.850 m** | 1.934 m | 0.792 |
| Epoch 30 (final) | 3.902 m | 1.934 m | 0.795 |

The best checkpoint beats the last, which is why selection is on validation RMSE. Validation
moved by 0.01 m over the final two epochs, so the schedule had converged; training longer is
not where the remaining error is.

Held-out GAMUS test (500-tile random subset, no TTA, no refinement) — see
`outputs/gamus_test.json` for the full breakdown:

| | RMSE | MAE | bias | NMAD | r |
|---|---|---|---|---|---|
| Heimdall | **5.42 m** | **2.46 m** | −1.48 m | 0.94 m | **0.751** |
| predict zero | 8.67 m | 4.39 m | −4.33 m | 0.50 m | — |

---

## VRAM requirements (measured, GTX 1080 / PyTorch 2.6)

Peak training memory is essentially linear in `batch × pixels`:
**peak ≈ 30 MiB + 17.4 GiB per megapixel-batch** (R² = 1.00).

| Crop | batch 1 | batch 2 | batch 4 | batch 8 |
|---|---|---|---|---|
| 256² | 1.2 GB | 2.3 GB | **4.6 GB** | 9.1 GB |
| 512² | 4.6 GB | 9.1 GB | 18.2 GB | 36.5 GB |

* **8 GB card (1080 / 3070)** → `--patch-size 256 --batch-size 4 --accumulate 8`
  (effective batch 32, ~4.6 GB). Mixed precision on Ampere roughly halves this, so
  `--batch-size 8` also fits there.
* **16 GB (T4 / V100)** → `--patch-size 256 --batch-size 8`, or 512² at batch 2.
* **24 GB+** → 512² at batch 4.
* The head is fully convolutional, so training on 256² crops and running inference on
  512² tiles is fine; only the receptive field per sample changes, not the weights.
* The head now uses GroupNorm, so small batches behave like large ones and batch 1 works.
* `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set for you: otherwise PyTorch sorts GPUs fastest-first
  and `--gpu 1` picks a different card than `nvidia-smi`'s GPU 1.

The **multiscale** head is cheaper than the affine one (it works mostly at feature
resolution): batch 4 @ 256² peaks at 1.9 GB and batch 2 @ 512² at 3.4 GB, measured on a
GTX 1080 in fp32.

**Inference needs 2.2 GB** (Depth Anything V2 Base + head, one 512² tile), so the deployed
pipeline runs on any modern GPU — or on CPU, more slowly.

---

## Reading the run

Each epoch logs `train loss | val RMSE | MAE | r`. Validation RMSE is what selects the best
checkpoint (`decoder_best_<tag>.pth`), not training loss. Expect it to fall steadily below
~4 m; if it plateaus early, train longer before changing anything. If it never drops below
the predict-zero baseline, something is wrong with the data pairing — check a few tiles from
`eval.py --save-examples`.

## Installing the result

```bash
cp checkpoints/decoder/decoder_best_v2.pth checkpoints/decoder/decoder_best.pth
git add -f checkpoints/decoder/decoder_best.pth
git commit -m "Head retrained on DA2 backbone" && git push
```

`decoder_best.pth` takes precedence over the legacy checkpoint automatically
(`heimdall/pipeline.py:DEFAULT_WEIGHTS`) and the backbone is read from the checkpoint, so no
other change is needed. Then refresh the numbers and the demo:

```bash
python scripts/benchmark_ahn.py --out outputs/benchmark_ahn     # paste into PROJECT_REPORT.md §4.4
python eval.py --weights checkpoints/decoder/decoder_best.pth --data-dir data/GAMUS --split test \
               --output outputs/gamus_test.json
python scripts/build_demo_site.py --site utrecht_suburb --size 600   # and veluwe_forest, valkenburg_hills
python scripts/publish_benchmarks.py                                 # refresh the Accuracy tab
```

## Tuning notes

* To weight a dataset more heavily, pass its directory twice.
* `--no-augment` disables flips/rotations/jitter — debugging only.
* The head is small (1.72 M parameters for the multiscale decoder). If it underfits badly,
  widen `hidden_dim` in `heimdall/decoder/multiscale_head.py` and retrain from scratch.
* Caching stores a **satellite-degraded** copy of every tile (down-sampled to ~0.5–1 m and
  back), and training samples it half the time, so the head also learns coarser sensors —
  closer to the imagery ISRO will supply at evaluation.
