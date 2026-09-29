# 02: Evaluation metrics suite

**What to build:** The metrics module used everywhere later: MAE, exact-match percentage, ±1 accuracy, and Spearman correlation per output (dev metrics); expected calibration error (ECE) for confidence calibration; and intraclass correlation (ICC) plus weighted Cohen's kappa for the AI-vs-expert validation study. Each metric is verified against hand-computed fixtures so the numbers in the paper can be trusted. This is pure math and can proceed in parallel with the model training ticket.

**Blocked by:** 01 (fixture dataset provides inputs and expected values for the metric tests)

**Status:** ready-for-agent

- [ ] Dev metrics (MAE, exact-match %, ±1 accuracy, Spearman) computed per output and match hand-computed fixtures
- [ ] ECE computed correctly and verified against a known-calibration fixture
- [ ] ICC and weighted Cohen's kappa computed correctly and verified against known-value fixtures
- [ ] Metrics module has no dependency on the trained model (works on any predicted-vs-actual arrays)