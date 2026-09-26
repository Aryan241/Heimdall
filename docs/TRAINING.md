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

**Batch size**: 8 GB cards fit about `--batch-size 4` at 512². If you hit `CUDA out of
memory`, halve it and raise `--accumulate` to keep the effective batch the same.

**Worth doing with three GPUs**: train two backbones at once and keep the better one —
`MODEL=vit-b TRAIN_GPU=1 TAG=v2b` alongside `MODEL=da3-mono-l TRAIN_GPU=0 TAG=mono`, then
compare their `benchmark_ahn` tables. They tie in the frozen-backbone test, but which one
trains better on GAMUS is an open question worth settling with evidence.

## Path B — Kaggle (if the workstation is unavailable)

GPU T4 ×2, Internet **on**. `/kaggle/working` is capped at 20 GB, which is why the compact
fetcher matters.

```python
!git clone https://github.com/Aryan241/Heimdall.git /kaggle/working/heimdall
%cd /kaggle/working/heimdall
!pip install -q rasterio scikit-image trimesh addict einops omegaconf timm "transformers>=4.44,<4.46" ultralytics

!python scripts/fetch_gamus.py --out /kaggle/working/GAMUS --splits train val --tiles 4000
!python scripts/build_ahn_dataset.py --out /kaggle/working/AHN --train 2000 --val 200 --workers 16

!python scripts/cache_backbone.py --dataset-dir /kaggle/working/GAMUS /kaggle/working/AHN --model vit-b
!python scripts/train_decoder.py --dataset-dir /kaggle/working/GAMUS /kaggle/working/AHN \
    --backbone-cache --model vit-b --epochs 30 --batch-size 8 --lr 3e-4 --num-workers 4 \
    --output-dir /kaggle/working/checkpoints --tag v2
```

If GAMUS is already attached as a read-only Kaggle input, point `--dataset-dir` at it and add
`--cache-root /kaggle/working/cache` to both the caching and training commands (`/kaggle/input`
cannot be written to).

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
python infer.py -i data/uploaded_image_1789657102996.tif -o heimdall-web/public/demo --name demo --tta
```

## Tuning notes

* To weight a dataset more heavily, pass its directory twice.
* `--no-augment` disables flips/rotations/jitter — debugging only.
* The head is small (1.4 M parameters). If it underfits badly, widen `hidden_dim` in
  `heimdall/decoder/head.py` and retrain from scratch.
* Caching stores a **satellite-degraded** copy of every tile (down-sampled to ~0.5–1 m and
  back), and training samples it half the time, so the head also learns coarser sensors —
  closer to the imagery ISRO will supply at evaluation.
