Status: ready-for-agent

# AI Suturing Quality Scoring — Spec

## Problem Statement

Suturing quality in medical training is assessed subjectively by faculty, with
no standardized, technology-driven protocol. This project builds an AI system
that objectively scores suturing photographs. The REU proposal promises
"AI-generated scores" validated by "comparing AI-generated scores with expert
clinician assessments" — the system must therefore produce scores that agree
with expert clinicians, plus a confidence signal, in a form suitable for a
peer-reviewed paper.

The dataset is image-only with per-image expert annotations on a 0–10 scale.
The expert-scored parameters are ISD (spacing consistency), Slack (thread
tightness), Position (placement appropriateness), Angulation (angle
consistency), Width (width consistency), and an Overall judgment combining
them. There is no stitch-level ground truth (no masks, keypoints, or boxes),
and the images vary wildly in resolution (383x549 to 1694x1406).

## Solution

A staged-conceptual, end-to-end-implemented (v1) PyTorch pipeline: a suturing
image is preprocessed, a multi-head ordinal-regression network scores ISD,
Slack, Position, Angulation, Width, and Overall on the native 0–10 scale; each
score carries a confidence derived from the model's ordinal distribution; the
Overall is a hybrid aggregation of the five parameters plus a learned residual,
calibrated against the annotated Overall. A CLI scores a folder of images to a
CSV, and a paper-artifacts generator produces tables and figures from one
command. The 206-image official Val split is used once for final numbers; the
432-image application cohort is held for external validation against expert
labels to be collected.

## User Stories

1. As a researcher, I want to score a folder of suturing images with one CLI
   command, so that I get per-image scores without manual effort.
2. As a researcher, I want the CLI output to be a CSV with one row per image,
   so that I can analyze results in any tool.
3. As a researcher, I want each image scored for ISD, Slack, Position,
   Angulation, and Width on 0–10, so that I can report parameter-level quality.
4. As a researcher, I want an Overall 0–10 score per image, so that I can
   report a single quality judgment.
5. As a researcher, I want a confidence value per score, so that I (and
   reviewers) can see how sure the model is.
6. As a researcher, I want the model trained on the Train cohort only, so that
   the official Val split stays uncontaminated for final evaluation.
7. As a researcher, I want model selection done on a held-out slice of Train,
   so that Val is touched exactly once.
8. As a researcher, I want the annotations loader to cast the text-typed score
   cells to integers, so that scores compare and sort correctly.
9. As a researcher, I want images joined to labels by filename, so that every
   image has exactly its own annotation.
10. As a researcher, I want preprocessing to normalize the inconsistent image
    resolutions, so that one model can score all images.
11. As a researcher, I want ordinal regression for each output, so that the
    ordered 0–10 nature of the scores is respected.
12. As a researcher, I want the Overall score computed from the five predicted
    parameters plus a learned residual, so that it is interpretable yet
    accurate.
13. As a researcher, I want the aggregation calibrated against the annotated
    Overall, so that the computed Overall matches expert judgment as closely
    as the data allows.
14. As a researcher, I want dev metrics (MAE, exact-match %, ±1 accuracy,
    Spearman) per output on held-out data, so that I can pick the best model.
15. As a researcher, I want confidence calibration reported (ECE), so that
    confidence values are trustworthy.
16. As a researcher, I want external validation metrics (ICC, weighted Cohen's
    kappa) between AI and expert scores on the application cohort, so that the
    paper has the promised AI-vs-clinician comparison.
17. As a researcher, I want per-trainee-level comparisons (2week, 4week,
    Resident) on the application cohort, so that the paper can discuss skill
    progression.
18. As a researcher, I want the pipeline to emit paper-ready tables and
    figures, so that results material is reproducible by one command.
19. As a researcher, I want fixed random seeds, so that results are
    reproducible.
20. As a researcher, I want a fixture-based test suite at the agreed seams, so
    that data-loading, aggregation, and metrics math are locked against
    regressions.

## Implementation Decisions

- **Framework**: PyTorch with torchvision pretrained backbones (ImageNet
  weights, fine-tuned).
- **Architecture (v1)**: end-to-end multi-head network, implicit geometry —
  one image in; five parameter heads plus an Overall head (ADR-0002). No
  explicit stitch-detection stage in v1: there is no stitch-level ground truth
  to supervise it, and resolution variance makes a measurement stage fragile.
- **Task type**: ordinal regression per output — each head outputs a
  distribution over scores 0–10 (CORAL-style or ordinal logit). Respects the
  ordered Likert-like scale; the distribution spread is the confidence signal.
- **Score scale**: native 0–10, exactly as annotated. Annotation cells are
  text strings — cast with int() at load; never compare/sort raw strings
  (lexicographic order misreports the range).
- **Preprocessing**: aspect-preserving resize to a fixed square input (default
  512, revisit if fine stitch detail is lost at higher resolutions) with
  per-channel normalization. Images range 383x549 to 1694x1406.
- **Overall**: hybrid aggregation — a function of the five predicted parameters
  plus a learned residual reading the image directly, calibrated against the
  annotated Overall (ADR-0001). A linear baseline reaches only R² ≈ 0.65
  (n=1216 pooled); the residual closes the gap. Fallback: if the residual
  fails to help, drop it and accept residual variance as clinician
  subjectivity.
- **Data discipline**: hold out ~10–15% of Train for model selection; official
  206-image Val used exactly once for final reported numbers; 432-image
  application cohort reserved for external validation.
- **Augmentation**: standard geometric (flips, small rotations) and photometric
  augmentations; transfer learning to compensate for n=1010.
- **Confidence**: per-output, from the ordinal distribution spread (epistemic
  uncertainty). Reported 0–1. Not annotated; model-generated.
- **CLI contract**: input = directory of PNGs; output = CSV with one row per
  image and columns for the five parameters, Overall, and confidence per
  output. Also scores the application cohort for the validation study.
- **Evaluation**: dev metrics MAE / exact-match % / ±1 accuracy / Spearman per
  output + ECE for confidence; validation metrics ICC and weighted Cohen's
  kappa (AI vs expert) on the application cohort.
- **Paper artifacts**: tables (per-output metrics; AI-vs-expert agreement) and
  figures (correlation scatter, calibration curves, trainee-level
  comparisons) generated by one pipeline command.
- **Reproducibility**: fixed seeds; documented preprocessing and training
  configuration.

## Testing Decisions

- A good test verifies external behavior, not implementation details: a loader
  that returns correct integers, an ordinal head whose distribution sums to 1,
  an aggregation that computes a known Overall from known parameters, metrics
  that match hand-computed fixtures.
- **Seams** (confirmed with the user):
  1. **CLI end-to-end** — run the pipeline on a tiny fixture dataset (a few
     images + fixture annotations); assert CSV schema (one row per image, six
     scores + confidence) and value ranges.
  2. **Annotations loader** — xlsx → rows with int-cast scores, joined to
     images on filename; locks the string-score gotcha.
  3. **Ordinal head contract** — logits → distribution over 0–10 summing to 1;
     argmax is the predicted score; spread is confidence.
  4. **Overall aggregation** — known five parameters in, known Overall out
     (function + residual).
  5. **Metrics** — ICC, weighted Cohen's kappa, ECE, ±1 accuracy against
     hand-computed fixtures.
- The training loop, backbone, and augmentation are not unit-tested; they are
  verified by the evaluation pipeline on held-out data.
- Prior art: none — this repo is net-new (data-only). Fixtures are
  hand-computed and synthetic.

## Out of Scope

- Explicit geometry stage (stitch detection, keypoint localization,
  segmentation) — possible future extension if stitch labels are ever
  collected (ADR-0002).
- Web UI / interactive scoring — pipeline CLI only.
- Per-person aggregation and reporting — filename suffix semantics are
  undocumented (`Image_NNNN_A_B_C_D`); outputs are per-image.
- Model deployment, serving, or integration into a teaching platform.
- The paper manuscript itself — this spec covers the system and its paper-ready
  artifacts, not the writing.
- Any modifications to the dataset or its splits.

## Further Notes

- Expert labels for the application cohort are to be collected; the validation
  tooling is built now and consumes them when they arrive, aligned to the
  `NNNN_Doc_XX_Itr_YY` filenames.
- GPU training is assumed for the paper's timelines; CPU training is possible
  but slower.
- The three cohort `.zip` archives were removed from the repo (extracted
  folders are authoritative); any future zip handling must filter
  `__MACOSX/`, `._*`, and `.DS_Store`.
- Domain vocabulary lives in `CONTEXT.md`; architectural decisions in
  `docs/adr/0001-*.md` and `docs/adr/0002-*.md`.