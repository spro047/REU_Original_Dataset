# 03: Ordinal regression model training

**What to build:** The trained scoring model: a single end-to-end multi-head network with implicit geometry (per ADR-0002), each head an ordinal regression over scores 0–10 (per the spec), a pretrained backbone fine-tuned on the Train cohort, standard geometric and photometric augmentation, a held-out slice of Train for model selection (the official Val split is touched only once, later), fixed random seeds, and checkpointing. Each output carries a confidence derived from the spread of its ordinal distribution. Dev metrics on the held-out slice are reported using the metrics module.

**Blocked by:** 01 (pipeline skeleton, fixtures, data layer)

**Status:** resolved

- [x] Model trains end-to-end from the Train cohort with the hold-out slice used for selection and the official Val untouched
- [x] Each head outputs a distribution over 0–10 (sums to 1); argmax is the predicted score; spread yields per-output confidence
- [x] Fixed seeds make training reproducible
- [x] Augmentation (geometric + photometric) and pretrained-backbone fine-tuning (`--pretrained`, resnet18) in place
- [x] Dev metrics (via the metrics module) reported on the held-out slice for model selection