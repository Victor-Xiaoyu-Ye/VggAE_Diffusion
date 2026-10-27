# Stage56 real cached-teacher diagnosis — 2026-10-27

The user supplied two real teacher caches at
`D:/中转站/teacher_diagnostic/case000/teacher.pt` and `case004/teacher.pt`,
with nonempty receipts. Both28,044,149-byte files pass size and SHA256 checks.
They include the original full9 and independent-prefix4 teacher outputs.
This supersedes the earlier limitation that only logs/empty local receipts were
available. No model weights, raw videos or new training were downloaded/run.

## Direct observations

There are576 queries perframe,14976 point/frame observations across the two
full/prefix pairs. Every observation fails BOTH existing score thresholds.

| Case/context | Maximum visibility | Maximum track confidence | Accepted observations |
| --- | ---: | ---: | ---: |
| case000 full9 | 0.041038 | 1.56214e-9 | 0/5184 |
| case000 prefix4 | 0.042095 | 1.60983e-9 | 0/2304 |
| case004 full9 | 0.028979 | 1.01151e-9 | 0/5184 |
| case004 prefix4 | 0.024378 | 8.10414e-10 | 0/2304 |

The thresholds were visibility>=0.5 and confidence>=0.2. Confidence minimum is
6.614e-11, with implied sigmoid logits roughly-23.44 to-20.25. Outputs are finite,
nonzero and variable, not placeholders or floating-point underflow. Frame0 query
coordinates exactly match all576 legal input points, yet every query is rejected.
Most later coordinates also remain inside the image. Thus neither ordinary
occlusion nor future out-of-bounds motion explains the score failure by itself.

Depth values vary, and rotations have determinants near1. These facts rule out a
teacher that returned only constants; they do NOT establish accurate geometry.
Raw unfiltered track overlays are saved on the matching lossy RAW preview columns.
They are qualitative diagnostics, not tracking ground truth or training labels.

Merely removing hard thresholds is not a repair. The raw visibility*confidence
weights are extremely small; reference transport multiplies source and destination
weights and divides by a denominator clamped to1e-8. Even a conservative bound on
the resulting reference-average scale is about1.64e-13/3.42e-14 for these cases
under a hypothetical removal of the thresholds. Changing qualification alone
therefore still leaves the reference feature numerically negligible. Arbitrarily
rescaling these scores would not establish reliable tracks either.

## Code audit and upstream lead

Local inference audit found no confirmed tuple-order, double-sigmoid, duplicate
RGB-normalization, token-layer selection or coordinate-scale error. Full9/prefix4
are multi-frame forwards; query_chunk_size=0. Head arithmetic isFP32, backbone
BF16. Tracking DPT uses pos_embed=False, so the Ascend depth positional-embedding
compatibility override does not affect tracking features. LF-normalized vendor
tracking files match the pinned upstream source hashes. All394 track-head keys
match checkpoint shapes; that does not certify their learned usefulness.

Official pinned [TrackLoss and DistillLoss](https://github.com/wzzheng/StreamVGGT/blob/05682bcb05683c4c71522e9cb618fc8ca349063c/src/dust3r/losses.py#L1181-L1195)
provide a concrete hypothesis: predicted confidence weights both coordinate and
visibility errors, without a term penalizing confidence collapse in TrackLoss.
Holding other outputs fixed, confidence approaching zero lowers this objective.
DistillLoss includes tracking, so do not claim this head was necessarily untrained.
The visibility loss also expects logits while the tracking head defaults to sigmoid
outputs. These source-level issues fit the symptoms but do NOT prove the released
checkpoint's training history or exclude an inference/device issue.

A scalar autograd demonstration keeps coordinate error and visibility unchanged
while confidence logit changes0 -> -5 -> -20. Loss falls0.799069 -> 0.010696 ->
approximately3.294e-9; the confidence-logit gradient remains positive. This only
demonstrates the objective's degeneracy, not a reproduction of checkpoint training.

## Decision and remaining discriminator

The current raw tracking scores are unusable as calibrated visibility/confidence
for this motion/reference conditioner. Do not lower thresholds or relaunch the
unchanged stage56 bank. The established local implementation/acceptance error was
treating a shape-compatible head as a validated teacher and accepting dataset-wide
zero support as a functioning motion/reference experiment.

This does not establish that StreamVGGT geometry features are unusable for our AE
or diffusion. Keep those components fixed during teacher diagnosis. A tracking
teacher can be replaced independently if needed; no new human labels are required.

The smallest useful follow-up uses the existing cluster checkpoint and one exact
RGB window plus a repeated-frame control, without optimization. Compare the
current BF16-backbone/FP32-head forward with FP32-backbone/FP32-head, verify
equivalence to the pinned multi-frame implementation, and record DPT feature
statistics, tracker normalized features, per-iteration logits, and predictor
weight/bias statistics. Do not use upstream single-frame cached inference as the
tracking reference: its tracker fixes the sole frame's coordinates to queries.
If matching full-precision/static-input runs retain collapsed scores, prioritize
another validated tracking teacher rather than spending AE/DiT training budget.
If precision or feature construction explains the divergence, fix that path and
regenerate controls under a new identity, then verify actual trajectories/transport
and matched condition sensitivity before a full generation comparison.

The two files are sufficient for the diagnosis reported here. Further cached
output files alone cannot reveal the intermediate activations or prove an exact
checkpoint-training cause. No production threshold, teacher or generator was
modified in this diagnosis.

## Evidence

All local analysis is under
`D:/workspace/VggAE_DataAudit/scene_control_20261027/diagnose_teacher/`:

- `real_teacher_audit.json`: verified input receipts, per-frame score statistics.
- `real_teacher_audit.case000.npz`, `.case001.npz`: compact tracks/scores.
- `score_failure_summary.json`: logs, filtering and reference-weight bounds.
- `track_coordinates_summary.json`: coordinate and depth diagnostic statistics.
- `case000_unfiltered_tracks.jpg`, `case004_unfiltered_tracks.jpg`: selected raw
  tracks over corresponding RAW preview frames0/3/8; no accuracy certification.
- `upstream_loss_degeneracy_demo.json`: scalar objective demonstration.

The existing CPU audit ran successfully on both real files. Numerical claims in
this document concern these two files; dataset-wide zero *effective* support was
established separately from the full stage56 logs. Do not generalize the exact
confidence distribution of two cases to all4096 training windows.
