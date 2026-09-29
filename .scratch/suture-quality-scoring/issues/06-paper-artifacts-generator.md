# 06: Paper artifacts generator

**What to build:** The one-command artifact generator for the paper: metrics tables (per-output MAE, exact-match %, ±1 accuracy, Spearman on the final Val numbers), AI-vs-expert agreement tables (ICC, weighted Cohen's kappa on the application cohort), correlation scatter plots of predicted vs expert scores, confidence calibration curves, and trainee-level comparison figures (2week vs 4week vs Resident). Everything a results section needs falls out of one command.

**Blocked by:** 05 (cohort scoring and validation outputs), 02 (metrics module)

**Status:** resolved

- [x] Metrics tables generated from final Val numbers (MAE, exact, ±1, Spearman, ECE per output; predictions rounded to the 0–10 scale for exact/calibration)
- [x] AI-vs-expert agreement tables generated from application-cohort scoring plus expert labels (per trainee level; demo labels shown until real ones arrive)
- [x] Correlation scatter plots, calibration curves, and trainee-level comparison figures emitted (per-output 2×3 grids + violin)
- [x] One command regenerates all artifacts from the scored CSVs (`artifacts` CLI subcommand; notebook 05)