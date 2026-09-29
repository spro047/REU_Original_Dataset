# AGENTS.md

Data-only repository (no code, no build) for the REU project
"Framework for Artificial Intelligence in Quality Control of Suturing Among
Undergraduates and Dermatology Residents". The only project documentation is
`Annexure 2_salma.docx` (proposal, work plan, references). Do not expect READMEs
or scripts — anything you build here (loaders, models, analysis) is net-new.

## Layout

```
Dataset/
  Train/Train_cohort/Train/         1010 PNGs + Train_annotations.xlsx
  Val/Validation_cohort/Validation/  206 PNGs + Validation_annotations.xlsx
  Test/Application_cohort/
    2week/                           160 PNGs  (Doc 01-10 x Itr 01-16)
    4week/                           160 PNGs  (Doc 01-10 x Itr 01-16)
    Resident/                        112 PNGs  (Doc 01-07 x Itr 01-16)
```

- `Test` is the unlabeled hold-out ("application") cohort; its subfolders encode
  trainee experience level (`2week`/`4week`/`Resident`). It has **no** annotations file.
- Train and Val are the labeled splits; each has a sibling `<Cohort>_annotations.xlsx`.

## Annotations (Train & Val xlsx, sheet "Sheet1")

Columns: `Name, Overall, ISD, Slack, Position, Angulation, Width` (all 0–9 scores).

- Column A is a 0-based index that equals the numeric id in the filename
  (verified: 0 mismatches in both files).
- All score cells are stored as **text strings**, not numbers — cast with
  `int()` before arithmetic. No missing values anywhere.
- Join images to labels on `Name` (filenames match directory contents 1:1 in
  both splits — verified both directions).

## Filename encodings

- Train/Val: `Image_NNNN_A_B_C_D.png` — `NNNN` is a unique 0-based id. The
  `A/B/C/D` suffix semantics are undocumented and the value sets differ between
  the Train and Val splits (e.g. part C is `{0,1,2}` in Train but
  `{0,1,2,3,4,6}` in Val). Do not assume a global coding scheme.
- Test: `NNNN_Doc_XX_Itr_YY.png` — `Doc` = document (1–10; Resident 1–7),
  `Itr` = iteration (1–16).

## Gotchas

- **The `.zip` archives (`Application_cohort.zip`, `Train_cohort.zip`,
  `Validation_cohort.zip`) contain a junk duplicate of every PNG under
  `__MACOSX/`** (e.g. Train zip = 1010 real + 1010 dupes). Never iterate a zip
  without filtering `__MACOSX/`, `._*` entries, and `.DS_Store`. The extracted
  folders are the clean source of truth and match the xlsx exactly.
- `Dataset/Test/Application_cohort/.DS_Store` is a macOS artifact — exclude it
  from any glob/scan.
- Images have inconsistent resolutions (observed 404x742 and 602x423). There is
  no preprocessing or resizing pipeline in the repo.

## Agent skills

### Issue tracker

Issues and specs live as markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.