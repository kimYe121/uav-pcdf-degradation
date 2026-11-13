# uav-modality-decay

English | [简体中文](README.zh-CN.md)

Diagnosing negative fusion under continuous modality decay in UAV RGB-IR oriented object detection.

## The question

Paired visible/infrared detectors are trained and benchmarked with both streams healthy. In deployment
one stream degrades — low light, lens contamination, misregistration — while the other remains usable.
Does fusion still beat simply discarding the bad stream, and where does it stop doing so?

```
NF(s) = mAP_fusion(s) − max(mAP_rgb_only(s), mAP_ir_only(s))
```

`NF < 0` means the fused model is worse than the better single modality. `s*` is the crossing point,
and the project asks whether `s*` moves with imaging conditions.

## Status

| Part | State |
|---|---|
| Pipeline (prepare → train → degrade → evaluate → curves) | working |
| Synthetic smoke test | passing |
| Real ATR-UMOD pilot | in progress |
| Reliability routing (method) | not implemented |

Synthetic runs exist only to verify that data conversion, training, degradation, metric computation
and plotting are wired correctly. They are not evidence for the research claim.

## Install

```bash
git clone https://github.com/kimYe121/uav-modality-decay.git
cd uav-modality-decay
pip install -e .
```

Python 3.10+. Training needs a CUDA-capable GPU; evaluation and the synthetic smoke test run on CPU.

## Quick start

```bash
python scripts/check_env.py      # torch / CUDA / ultralytics sanity check
python scripts/run_smoke.py      # synthetic end-to-end run, a few minutes
```

Outputs: `results/*_results.csv`, `results/*_summary.txt`, `results/*_curves.png`.

## Data

The experiments use [ATR-UMOD](https://github.com/supercpy/ATR-UMOD) (ICCV 2025): 13,353 registered
RGB–IR pairs, oriented boxes, 11 vehicle classes, six per-image condition attributes. Download it
from the link in that repository (the paper's Baidu Pan code is `accd`) and unpack to:

```
data/ATR-UMOD/
├── train/
│   ├── images/       # visible
│   ├── images_ir/    # infrared (grayscale)
│   ├── labels/       # VOC-style XML, visible
│   └── labels_ir/    # VOC-style XML, infrared
└── test/
    └── (same structure)
```

Convert to YOLO-OBB and carve a validation split by scene:

```bash
python scripts/prepare_atr.py check   --root data/ATR-UMOD
python scripts/prepare_atr.py convert --root data/ATR-UMOD --out data/atr_umod
```

`check` prints class names, instance counts, coordinate arity and attribute fields before anything is
written. Do not split by random frame: adjacent frames from one flight are near-duplicates and inflate
validation scores. Report mean ± std over at least three seeds.

## Degradation protocols

| Protocol | Operation | What stays intact |
|---|---|---|
| P1 contamination | low light + read noise, fog scattering, blur, or additive noise on RGB | IR untouched |
| P2 zeroing | RGB driven toward black | IR untouched |
| P3 misregistration | IR frame translated by 2–20 px | both images individually clean |

P1/P2 reduce single-modality quality. P3 keeps each stream's own quality at 1.0 and only breaks
cross-modal correspondence — the case a single-modality quality score cannot catch.

## Model views

| View | Input |
|---|---|
| `rgb` | visible only |
| `ir` | infrared only (grayscale replicated to three channels) |
| `fused` | RGB with IR injected into the B channel (channel-concat proxy) |

The fusion proxy is intentionally minimal: it cannot reweight a degraded stream, which is the failure
mode under study. A learned two-stream baseline replaces it in the full experiments. Pixel averaging
was tried first and abandoned — it preserves the clean stream's information and hides the effect,
which locates the phenomenon at the feature level rather than the pixel level.

## Metrics

- `RA(s) = mAP(s) / mAP(clean)` — resistance ability, following MultiCorrupt
- `NF(s)` — negative fusion gap, defined above
- `s*` — first sign change of `NF`, linear interpolation between levels
- slope of `mAP` against `s` — the weaker claim when no crossing occurs

## Project layout

```
configs/                 experiment configs (smoke, pilot)
scripts/                 entry points: env check, synthetic build, data prep, smoke run
src/modality_decay/      package: synth, degrade, datasets, train, evaluate, metrics, plots
tools/                   standalone conversion and metric helpers
```

## References

- ATR-UMOD / PCDF, ICCV 2025 — dataset and condition-aware fusion baseline
- M²D-LIF, ICCV 2025 — names fusion degradation
- RSC-MD, arXiv 2511.15433 — gradient-suppression account of fusion degradation
- MultiCorrupt, arXiv 2402.11677 — RA / RRA corruption-robustness metrics
- Ultra-Fusion, arXiv 2606.21223 — factor-wise reliability scheduling in SLAM
- DRNet — dynamic routing under missing modalities
