# 01: Pipeline skeleton and data foundation

**What to build:** The vertical skeleton of the scoring pipeline: a fixture dataset (a handful of real suturing images plus a fixture annotations file), the annotations loader (reads the xlsx, casts the text-typed score cells to integers, joins images to labels by filename), preprocessing (aspect-preserving resize to a fixed square input, per-channel normalization), a minimal end-to-end scoring path using a toy model, and a skeleton CLI that scores a folder of images and writes a CSV with one row per image and columns for ISD, Slack, Position, Angulation, Width, Overall, and confidence. This makes the full path from images to CSV work end-to-end for the first time, on fixtures only.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Fixture dataset exists (small number of real images + fixture annotations) and is reproducible
- [x] Annotations loader returns integer scores on the native 0–10 scale (never raw text) and joins images to labels by filename with zero mismatches
- [x] Preprocessing normalizes wildly inconsistent input resolutions to one fixed input size
- [x] CLI accepts an image directory and writes a valid CSV (one row per image, six scores + confidence per output)
- [x] Tests green at the loader, preprocessing, and CLI end-to-end seams