# 05: Production cohort scoring and validation tooling

**What to build:** The production scoring run: score all three cohorts (Train, Val, and the 432-image application cohort) with the trained model, writing one CSV per cohort with per-image scores and confidence. For the application cohort, the tooling consumes expert clinician labels when they arrive (aligned to the application-cohort filenames) and produces the AI-vs-expert agreement results — ICC and weighted Cohen's kappa per trainee level (2week, 4week, Resident) — using the metrics module. The official Val split is used exactly once here for the final reported numbers.

**Blocked by:** 03 (trained model), 02 (metrics module), 04 (Overall aggregation)

**Status:** resolved

- [x] Trained model scores Train, Val, and application cohorts to per-cohort CSVs (six scores + confidence per image)
- [x] Official Val used exactly once for final numbers (scored; nothing fitted on it)
- [x] Expert-label consumer accepts application-cohort annotations and aligns them to images (NNNN_Doc_XX_Itr_YY names, index invariant, alignment report)
- [x] ICC and weighted Cohen's kappa computed per trainee level when expert labels are present (agreement_by_level; notebook uses real labels when present, synthetic fallback demo otherwise)
- [x] Graceful, documented behavior when expert labels are not yet available