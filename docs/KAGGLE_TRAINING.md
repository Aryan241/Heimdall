# Retraining the decoder head on Kaggle

**Why this is needed.** The shipped head was trained on a *Depth Anything V3 Metric* backbone.
Measured against airborne LiDAR (`scripts/benchmark_ahn.py`, 8 Dutch sites, none used for
training), that backbone is nearly blind to height in straight-down imagery:

| Backbone | Mean correlation with LiDAR height |
|---|---|
| DA3 Metric-Large (old default) | **0.06** |
| DA2 Small / Base / Large | 0.35 / **0.41** / 0.41 |

With no height signal to work from, the head could only guess from colour and texture:
overall nDSM RMSE **7.44 m** versus **8.37 m** for predicting zero everywhere. The fix is to
retrain the head on a **Depth Anything V2** backbone (`vit-b` is the default: as accurate as
`vit-l` here and 3× faster).

Training must run on a GPU. It is impractical on Apple Silicon because the backward pass
through the head's dilated (ASPP) convolutions is pathologically slow on MPS.

---

## What you need

* A Kaggle notebook with **GPU T4 ×2** (one is used) and **Internet: On**.
* Your GAMUS dataset attached as a Kaggle input (optional but recommended).
* ~2 hours of session time.

## Step 1 — Notebook setup

```python
!git clone https://github.com/Aryan241/Heimdall.git /kaggle/working/heimdall
%cd /kaggle/working/heimdall
!pip install -q rasterio scikit-image trimesh addict einops omegaconf timm "transformers>=4.44,<4.46" ultralytics
```

## Step 2 — Build the LiDAR-labelled training set (~30–40 min)

Aerial photos + AHN LiDAR heights from the Dutch national services, sampled across the
country. Tiles within 3 km of any benchmark site are excluded, so the benchmark stays honest.

```python
!python scripts/build_ahn_dataset.py --out /kaggle/working/AHN --train 3000 --val 300 --workers 16
```

Add GAMUS too if you have it attached — more variety is better. GAMUS tiles are 1024², AHN
tiles 512²; both are handled.

## Step 3 — Cache the frozen backbone (~10 min on T4)

The backbone never changes during training, so run it once per tile. This also stores a
**satellite-degraded** copy of every tile (down-sampled to ~0.5–1 m and back), so the head
learns coarser sensors too — closer to what ISRO will supply.

```python
!python scripts/cache_backbone.py \
    --dataset-dir /kaggle/working/AHN /kaggle/input/<your-gamus-folder> \
    --cache-root /kaggle/working/cache --model vit-b
```

`--cache-root` is required for `/kaggle/input`, which is read-only.

## Step 4 — Train (~1 h for 30 epochs)

```python
!python scripts/train_decoder.py \
    --dataset-dir /kaggle/working/AHN /kaggle/input/<your-gamus-folder> \
    --cache-root /kaggle/working/cache --backbone-cache --model vit-b \
    --epochs 30 --batch-size 8 --lr 3e-4 --num-workers 4 \
    --output-dir /kaggle/working/checkpoints --tag v2
```

Do **not** pass `--resume` with the old checkpoint: it was fitted to a different backbone's
output, so it starts you in the wrong place.

Watch the per-epoch line: `val RMSE` should fall below ~4 m and keep dropping. The best
checkpoint by validation RMSE is saved as `decoder_best_v2.pth`.

## Step 5 — Check it against LiDAR (~15 min)

```python
!python scripts/benchmark_ahn.py --out /kaggle/working/bench   # add --gsd 0.5 1.0 for satellite-like tests
```

This prints the table to paste into `PROJECT_REPORT.md` §4.4. Compare against the current
baseline (mean nDSM RMSE 7.44 m, predict-zero 8.37 m). If you also have GAMUS:

```python
!python eval.py --weights /kaggle/working/checkpoints/decoder_best_v2.pth \
    --data-dir /kaggle/input/<your-gamus-folder> --split test --output /kaggle/working/gamus_test.json
```

## Step 6 — Bring the checkpoint back

Download `decoder_best_v2.pth` from the notebook output, then in the repo:

```bash
cp ~/Downloads/decoder_best_v2.pth checkpoints/decoder/decoder_best.pth
git add -f checkpoints/decoder/decoder_best.pth && git commit -m "Head retrained on DA2 backbone" && git push
```

`decoder_best.pth` is picked up automatically in preference to the legacy checkpoint
(`heimdall/pipeline.py:DEFAULT_WEIGHTS`), and the backbone is read from the checkpoint, so
nothing else needs changing. Re-run `scripts/benchmark_ahn.py` locally to confirm, then
regenerate the demo scene:

```bash
python infer.py -i data/uploaded_image_1789657102996.tif -o heimdall-web/public/demo --name demo --tta
```

---

## Tuning notes

* `--batch-size 8` fits comfortably on a T4 at 512²; raise it if memory allows.
* If validation RMSE plateaus early, train longer (`--epochs 60`) before changing the model.
* To weight a dataset more heavily, pass its directory twice.
* `--no-augment` disables flips/rotations/jitter — only useful for debugging.
* The head is small (1.4 M parameters); if it underfits badly, widen it in
  `heimdall/decoder/head.py` (`hidden_dim`) and retrain from scratch.
