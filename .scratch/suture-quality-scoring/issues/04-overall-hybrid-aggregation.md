# 04: Overall hybrid aggregation

**What to build:** The aggregation stage that turns the five predicted parameters (ISD, Slack, Position, Angulation, Width) into the Overall score: a learned function of the five parameters plus a residual term that reads the image directly to close the gap a linear baseline leaves (R² ≈ 0.65), calibrated against the annotated Overall (per ADR-0001). The aggregation is testable in isolation: known parameters in, known Overall out. If the residual fails to improve validation performance, drop it and accept residual variance (documented fallback).

**Blocked by:** 03 (needs the trained model's parameter outputs to calibrate against)

**Status:** resolved

- [x] Aggregation computes Overall from the five predicted parameters, verified against hand-computed inputs
- [x] Learned residual term is trained and evaluated against the fallback (no residual) — K-fold out-of-fold MAE
- [x] Calibrated against the annotated Overall on held-out data
- [x] Decision on residual-vs-fallback recorded (with_residual won: MAE 1.011 vs 1.125 on the held-out slice; rerun with the paper model to lock for the manuscript)