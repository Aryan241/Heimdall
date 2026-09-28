# Licensing and attribution

Every third-party model and dataset Heimdall uses, with the licence each actually declares
(checked against the upstream model cards and package metadata, not from memory). Two of them
carry obligations that matter for how this software can be used, and both are flagged below
rather than buried.

## Models

| Component | Used for | Licence | Consequence |
|---|---|---|---|
| **Depth Anything V2 Base** | the default backbone | **CC-BY-NC-4.0** | **Non-commercial only.** Research, evaluation and academic use are fine; a commercial deployment is not. |
| Depth Anything V2 Small | selectable (`--model vit-s`) | Apache-2.0 | permissive |
| Depth Anything V2 Large | selectable (`--model vit-l`) | CC-BY-NC-4.0 | non-commercial |
| **Depth Anything V3 Mono Large** | selectable (`--model da3-mono-l`) | **Apache-2.0** | permissive |
| SegFormer B0 (ADE20K) | optional ground mask | NVIDIA licence ("other") | check before redistributing |
| **YOLOv8n / YOLOv8n-OBB** (Ultralytics) | vehicle ground prior, GSD heuristic | **AGPL-3.0** | strong copyleft — see below |

## Data

| Dataset | Used for | Licence |
|---|---|---|
| **GAMUS** (earthflow) | training and the held-out test split | CC-BY-4.0 — attribution required |
| **AHN** via PDOK | training tiles and the LiDAR benchmark | Dutch open government data, public domain |
| **PDOK aerial orthophotos** | benchmark and demo imagery | Dutch open government data |
| **Copernicus GLO-30 DEM** | automatic terrain datum | free, ESA/Copernicus attribution required |
| CartoDEM (ISRO/NRSC) | optional terrain datum | per Bhoonidhi terms; not redistributed here |

## The two that constrain use

### The default backbone is non-commercial

Depth Anything V2 **Base** — what the shipped checkpoint is trained against — is
**CC-BY-NC-4.0**. For a competition entry, academic work or an internal ISRO evaluation this is
not a problem. For an operational or commercial deployment it is.

There is a measured way out. The backbone comparison in the report (§4.4) found **DA3 Mono
Large statistically indistinguishable from DA2 Base** — median correlation with LiDAR height
0.58 against 0.56 — and DA3 Mono is **Apache-2.0**. The costs are that it is roughly three
times slower per tile, and the decoder head is fitted to one backbone's output distribution, so
switching means retraining the head.

So a permissively licensed build is available at a known price, and the evidence for it is
already in the report.

### The vehicle detector is AGPL-3.0

Ultralytics YOLO is **AGPL-3.0**, a strong copyleft licence: distributing software that
incorporates it generally requires releasing that software under the AGPL as well, including
over a network interface.

It is used in two optional places — the vehicle ground prior in `heimdall/refine/vehicles.py`
and the GSD heuristic in `heimdall/calibration/heuristic.py`. Both degrade gracefully when it is
absent: the pipeline logs that the detector is unavailable and continues. Running with
`--no-refine`, or simply without the weights present, removes the dependency at the cost of the
parking-lot correction, which is worth under 0.03 m of accuracy (§4.4) and some visual tidiness.

## Attribution

Any publication or deployment should credit:

* **GAMUS** — Earthflow, CC-BY-4.0.
* **AHN** and the aerial orthophotos — PDOK / Netherlands open government data.
* **Copernicus GLO-30** — "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and
  © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and
  ESA, all rights reserved."
* **Depth Anything V2 / V3** — the respective authors, under the licences above.

## Heimdall's own code

The code in `heimdall/`, `scripts/`, `heimdall-web/` and the top-level entry points is original
work. **No licence has been chosen for it yet** — that is a decision for the project owners, and
it interacts with the AGPL dependency above, so it should be made deliberately rather than by
dropping in a default.
