# uav-modality-decay

Diagnosing negative fusion under continuous modality decay in UAV RGB-IR oriented object detection.

Given a paired visible/infrared detector, this repository asks a narrow question:

> When one modality decays continuously while the other stays clean, at what point does
> fixed-weight fusion fall **below the better single modality** — and does that crossing
> point move with imaging conditions?

`NF(s) = mAP_fusion(s) − max(mAP_rgb_only(s), mAP_ir_only(s))`, negative `NF` means the fused
model is worse than simply discarding the degraded stream. `s*` is the crossing point.

## Status

| Part | State |
|---|---|
| Pipeline (train → degrade → evaluate → curve) | working on synthetic data |
| Real ATR-UMOD integration | pending dataset download |
| Reliability gating (method) | not implemented yet |

**Synthetic results are a code-correctness check, not evidence.** The synthetic generator is
deliberately built so that clean fusion beats both single modalities; it exists to verify that
data conversion, training, degradation, metric computation and plotting are wired correctly.

## Layout

```
configs/pilot.yaml          experiment config
scripts/check_env.py        verify torch/CUDA/ultralytics
scripts/build_synth.py      generate synthetic RGB-IR pairs + YOLO-OBB labels
scripts/run_smoke.py        full smoke run
src/modality_decay/
  synth.py                  synthetic pair generator
  degrade.py                P1 contamination / P2 zeroing / P3 misregistration
  datasets.py               YOLO data.yaml + degraded split materialization
  train.py                  one detector per view (rgb / ir / fused)
  evaluate.py               validation wrapper
  metrics.py                RA / NF / s* / slope
  plots.py                  availability-performance curves
tools/                      dataset conversion helpers
```

## Quick start

Use the existing conda environment (torch 2.11 + cu128 in this setup). Do not create a second
environment; install into the current one if a package is missing.

```bash
D:/anaconda3/python.exe scripts/check_env.py
D:/anaconda3/python.exe scripts/run_smoke.py
```

Outputs land in `results/` (`*_results.csv`, `*_summary.txt`, `*_curves.png`).

## Three degradation protocols

| Protocol | Operation | What stays intact |
|---|---|---|
| P1 contamination | low light + read noise, fog scattering, blur, or additive noise on RGB | IR frame untouched |
| P2 zeroing | RGB scaled toward black | IR frame untouched |
| P3 misregistration | IR frame translated (2–20 px) | both images individually clean |

P1/P2 drive single-modality quality down. P3 keeps `q_self` at 1.0 for both streams and only
breaks cross-modal correspondence — the case a single-modality quality score cannot catch.

## Views trained

| View | Input |
|---|---|
| `rgb` | visible only |
| `ir` | infrared only (grayscale replicated to 3 channels) |
| `fused` | fixed-weight input fusion `0.5*rgb + 0.5*ir` |

The fixed-weight fusion is a pilot proxy. It reproduces the failure mode of interest (a fusion
model whose weighting was learned under "both streams healthy") at near-zero implementation cost.
A learned two-stream baseline replaces it for the real experiments.

## Real data

Download ATR-UMOD, then convert to YOLO-OBB and split by scene:

```bash
D:/anaconda3/python.exe tools/atr_umod_prepare.py check --root <ATR-UMOD-root>
D:/anaconda3/python.exe tools/atr_umod_prepare.py convert --root <ATR-UMOD-root> --out data/atr_umod
```

Never split frames randomly: adjacent frames from one flight are near-duplicates, which inflates
validation scores. Split by scene and report mean±std over at least three seeds.

## Notes on naming

`availability-aware fusion` is already used by ASF (NeurIPS 2025), and `dynamic routing for
modality missing` by DRNet. This project deliberately uses **modality decay** to mark the
difference: continuous decay with a measurable crossing point, not binary missing-modality routing.
