# v1 architecture: end-to-end multi-head CNN with ordinal regression

The first version of the scoring model is a single end-to-end network with
implicit geometry: one image in, five parameter heads (ISD, Slack, Position,
Angulation, Width) plus an Overall head, each trained with ordinal regression
(a distribution over 0–10) rather than plain regression or classification.
Explicit geometry (a separate stage that detects stitches and measures
spacing/angle/width) is deliberately NOT built for v1, even though the project
objective describes "analyzing stitch geometry and technique": the dataset has
no stitch-level labels (no masks, keypoints, or boxes), so an explicit stage
could not be supervised, and the images vary wildly in resolution (383x549 to
1694x1406 observed), making a fragile measurement stage a worse bet than a
network that learns geometry implicitly. The per-stitch consistency semantics
of the parameters ("whether the spacing is consistent") are learned rather than
computed. Explicit geometry remains a possible research extension if stitch
labels are ever collected; ordinal regression is kept because the scores are
ordered Likert-style ratings and the distribution spread doubles as the
model's confidence signal.