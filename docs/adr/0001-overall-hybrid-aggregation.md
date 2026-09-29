# Overall score as hybrid aggregation with a learned residual

The final `Overall` score (0–10) is computed by an aggregation stage over the
five predicted quality parameters (ISD, Slack, Position, Angulation, Width),
calibrated against the expert-annotated `Overall`. Because a linear regression
of `Overall` on the five parameters reaches only R² ≈ 0.65 (n=1216 pooled),
leaving roughly a third of the variance unexplained by the parameters alone,
the aggregation also includes a learned residual term that reads the image
directly to close the gap. This keeps the score interpretable (the parameters
are the backbone) while not handcuffing accuracy to what the five parameters
can express. Fallback: if the residual term fails to improve validation
performance, drop it and accept the residual variance as irreducible
clinician subjectivity.