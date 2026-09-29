# Suturing Quality Assessment

A research project that builds an AI system to objectively assess suturing
skill from photographs of sutured wounds/pads, replacing subjective faculty
evaluation. The system analyzes stitch geometry and technique, evaluates
quality parameters, and produces an overall quality score for the person who
performed the suturing.

## Language

**Suture quality score**:
A 0–10 rating assigned by expert clinicians to an image's suturing, per
parameter and overall. The top score 10 is rare in the data.
_Avoid_: grade, mark, rating

**ISD**:
Whether the spacing between stitches is consistent.
_Avoid_: spacing, inter-suture distance

**Slack**:
Whether the thread appears appropriately tight or loose.

**Position**:
Whether the stitch is placed appropriately on the wound/pad.

**Angulation**:
Whether the stitch angle is consistent.

**Width**:
Whether the stitch width is consistent.

**Overall**:
The expert's per-image judgment combining the five quality parameters into a
single score. The final presentation scale for the person-level evaluation is
under decision (annotated values run 0–10; 10 is rare).
_Avoid_: total, final grade

**Confidence**:
The model's self-reported certainty in a prediction, derived from its
epistemic uncertainty (the spread of the predicted score distribution over
0–10). Generated per prediction and per output (each parameter and Overall);
there is no annotated confidence in the dataset.

**Cohort**:
One of the three data splits: Train, Val, or the application (Test) cohort.
Train and Val are labeled; the application cohort is not.
_Avoid_: split, dataset (when meaning a specific cohort)

**Application cohort**:
The unlabeled hold-out cohort of suturing images, grouped by trainee
experience level (`2week`/`4week`/`Resident`). Used for external validation by
comparing AI-generated scores with expert clinician assessments.
_Avoid_: test set, hold-out

**Trainee level**:
The experience level encoded by an application-cohort subfolder: `2week`,
`4week`, or `Resident`.