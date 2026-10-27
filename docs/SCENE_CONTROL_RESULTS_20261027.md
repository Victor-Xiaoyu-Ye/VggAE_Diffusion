# Stage56 local result review — 2026-10-27

The jobs completed, but the intended motion/reference comparison did not operate.
All logged B/C training batches and all evaluation records have zero valid track
support. C never received nonzero transported appearance in the logged batches.
Camera conditions and their learned injection were active. These results cannot
establish that sparse motion control or appearance transport fails to help.

Update: the user subsequently supplied two real teacher payloads. See
[the verified cached-teacher diagnosis](SCENE_TRACK_TEACHER_FAILURE_20261027.md).
Both tracking scores are extremely low before our filtering, including frame0;
merely relaxing cutoffs cannot restore useful confidence-weighted reference input.
Earlier missing-teacher limitations below describe the original logs-only review.

## Evidence and scope

User supplied `D:/中转站/scale`; no OBS/network access or raw-data download was used.
Reviewed full and smoke `scene_street_control_reference_v1*` directories, matched
metadata, 24-rank training records, teacher setup, publication receipts, and videos.
Analysis artifacts are outside Git under:

`D:/workspace/VggAE_DataAudit/scene_control_20261027/`

- `integrity/integrity.json`: completion, data counts, hashes, complete MP4 decoding.
- `metrics/stage56_metrics.json`: matched metrics and control activation records.
- `visual_review/`: all50 final-frame comparisons, matched person timelines/ROIs.
- `visual_review/case_mapping.json`: local case to original evaluation index mapping.

The local download contains no actual teacher, control-cache or checkpoint `.pt`
tensors. Checkpoint receipts exist; this review does not verify weight payload
integrity or fresh-process restore. Some local teacher receipt placeholders are
empty, so they do not expose visibility/confidence distributions.

## Completion and resources

All three full arms completed5000 optimizer updates on24ranks. Each reports
`optimization_plateau_requires_visual_review`, `evaluation_status=completed`.
The monitor's best probe occurred at1000, followed by4 stale checks. This is a
supervised optimization stopping rule, not proof of visual quality convergence.

| Arm | Recorded total hours | Training hours | Median DI | Peak allocated memory |
| --- | ---: | ---: | ---: | ---: |
| A baseline | 7.080 | 6.943 | 1411.9 | 25.40 GiB |
| B motion | 7.104 | 6.978 | 1373.4 | 25.40 GiB |
| C reference | 7.284 | 7.147 | 1362.6 | 25.40 GiB |

DI here is future target tokens/second/NPU; it is not videos/second. All full/smoke
launcher nodes exited0. No full-run OOM, NaN, traceback or idle-reclamation evidence.
Smoke node2 initially timed out connecting TCPStore, then retried and completed.
All smoke arms completed3 updates; its zero-support problem was already present.

All1344 MP4 files fully decode to9frames. All4032 non-PT receipt-listed artifacts
match size and SHA256. All192 publication status records report both destinations
successful. This establishes the downloaded media/log integrity, not remote
checkpoint readability now. Each full arm has428 videos:100 source,32 per monitor
at1000/2000/3000/4000, and200 final.

## Data and fair comparison

Training preparation retained4096/4096 windows in1861 prepared shards. All24
teacher workers completed. Evaluation retained50/64 cases; the following original
indices were skipped because RGB replay differed from sealed evaluation frames:

`5,7,9,14,15,20,21,24,32,33,41,42,43,48`

This is a replay-consistency failure, not evidence that these videos are missing.
Capture replay difference magnitudes and frame provenance before changing its
tolerance. Do not silently relax the check or feed controls aligned to wrong RGB.
The surviving50 are identical across source/A/B/C, including IDs, frame_ids,
sampling seeds/noise and observed4/generated5. Final evaluation is50cases ×2seeds
×online/EMA =200 outputs perarm. Do not compare these means with old64-case means.

## What actually entered the model

Across12024 logged training records perarm (24ranks,501 logged steps):

| Signal | B | C |
| --- | --- | --- |
| `control_track_support_fraction` | always0 | always0 |
| `control_reference_support_fraction` / mean | always0 | always0 |
| `control_reference_rms` | always0 | always0 |
| `control_motion_rms` | always0.3535533845 | always0.3535533845 |
| Camera input RMS | nonzero, about0.612–0.619 | nonzero, about0.612–0.619 |
| Learned control injection | active | active |

Motion RMS equals sqrt(1/8), consistent with only the UNKNOWN channel being1.
Every saved evaluation sample at source and1000–5000 has `track_valid_fraction=0`.
The trained injection reaches about0.02766 RMS, up to0.0443 of the base input RMS.
Thus this is not a completely disconnected control MLP. It is a camera-plus-constant
unknown condition experiment, with no effective motion/reference contrast.

The experiment names and declared condition policies describe the intended task;
they are not evidence of actual control coverage. C versus B cannot measure an
appearance-transport contribution when its input is zero.

## Failure localization and uncertainty

`utils/scene_control_data.py::_probability_tracks` accepts a track only when
visibility>=0.5, track_confidence>=0.2, finite coordinates and in-image coordinates
all pass. This weight is computed before camera-compensated dynamic/static labels.
Therefore later geometric residual thresholds cannot explain zero raw track weights.

The pinned tracker forces frame0 coordinates back to the valid query grid every
iteration. Even frame0 has zero accepted tracks, pointing to rejection by its
visibility/confidence qualification. It cannot be explained solely by future
occlusion, camera motion or future-coordinate drift. The wrapper does not apply a
second sigmoid; the track head applies it once. All24 worker reports match all394
track-head keys (65,941,396 parameters), with no missing or shape-mismatched keys.
This rules out an absent/randomly initialized head, but not ineffective weights
inside the supplied checkpoint or an upstream numerical/input issue.

The LF-normalized local teacher source identity exactly matches the run's
`ec2411168f0cd876233b0d0e4511371295dd99ca561bce3f10bcc7762571ece9`.
This localization corresponds to the code actually run. The missing teacher
tensors prevent distinguishing visibility collapse from confidence collapse or
identifying why it happened. Do not guess that reducing thresholds will fix it.

The validation gap is established: `validate_controls` checks shapes, finite
values and ranges, and allows zero coverage. This is appropriate for individual
unsupported windows, but the smoke acceptance did not distinguish dataset-wide
absence of the central condition from a functioning motion/reference experiment.
Execution success was incorrectly sufficient to proceed to the full comparison.

## Quality results

Paired future-frame RGB L1, lower is closer to the one recorded future:

| Model | Online | EMA |
| --- | ---: | ---: |
| Source stage53 step31069 | 0.081693 | — |
| Stage54 control6000, matched50 | — | 0.081030 |
| A baseline5000 | 0.078032 | 0.078187 |
| B5000, actual camera/unknown only | 0.077715 | 0.077844 |
| C5000, actual camera/unknown only | 0.077714 | 0.077855 |
| AE reconstruction | 0.036765 | — |
| Copy last observed RGB | 0.101427 | — |

Most source-to-final gain also occurs in A: EMA improves4.29% from source, versus
4.71% for B and4.70% for C. B/C improve only about0.4% relative to A. Paired
case-mean EMA B−A=-0.0003432 (37/50 wins), C−A=-0.0003327 (38/50 wins).
Case bootstrap95% intervals are [-0.0005035,-0.0001923] and
[-0.0004857,-0.0001887], conditional on these specific trained runs; they do not
measure variability across independent training seeds. C−B=+0.00001048, interval
[-0.00001474,+0.00003945]. No distinct reference benefit is established.
Against the previous stage54 control6000 EMA on exactly these50 cases, C improves
RGB L1 about3.9% (40/50 wins), but its motion amplitude and denoising probes are
worse. That older model's motion ratio is0.8117 and probe0.045229/0.057656.

The future RGB-motion ratio (ratio of mean generated to GT temporal L1) decreases
from source0.8150 to A/B/C EMA0.7826/0.7837/0.7841. This proxy also responds to
camera, exposure and artifacts; it is not object motion correctness. Full50-case
unweighted x0 probe worsens: source global/change MSE0.045595/0.058103 versus
A EMA0.049112/0.063150 and C EMA0.048839/0.062764. Lower sampled RGB L1 therefore
does not imply uniformly better denoising or dynamics. These are not FVD or
geometry benchmarks, and alternate valid futures can differ from recorded GT.

Visual inspection covered final seed101 for all50 cases and sequential person
examples source004/011. Buildings/street layouts remain recognizable; some
background alignments improve. People still have doubled contours, translucent
limbs and disappearing detail, especially in later future frames. A/B/C are
visually very similar on these failures. The AE keeps cleaner person silhouettes,
so a substantial generation-to-reconstruction gap remains. No visual breakthrough
or best-overall-checkpoint replacement is established. Retain historical stage53
as a reference; preserve stage56 as an additional baseline/camera pilot.

## Next decision

Do not spend another full run on the unchanged prepared control bank. Do not
interpret this run as evidence against reference guidance or the overall VGGT
generation direction. The next useful work is a small read-only teacher diagnosis:

1. Retrieve a few saved full/prefix teacher tensors; record per-frame visibility
   and confidence quantiles, their separate/joint acceptance, and coordinate bounds.
   Start with case000 and the people case004; expand to8–16 varied windows.
2. Overlay raw tracks and accepted tracks on corresponding frames. Verify useful
   correspondence, not merely positive counts. Distinguish checkpoint/head quality,
   preprocessing and precision issues before changing qualification rules.
3. Record these small statistics in mirrored JSON so logs alone reveal future
   failures. Explicitly report a camera-only fallback if motion is unusable;
   an unsupported individual window should not stop unrelated baseline training.
4. After obtaining useful tracks, validate transported observed appearance and
   condition sensitivity on a small matched set before another full comparison.
   Use supplied versus shuffled/zero controls with the same samples/noise. Keep
   future appearance leakage prohibited. This is a functional test, not a quality
   claim from a tiny overfit subset.

Two saved teacher object paths relative to the controls run are:

```
prepare/cases/5a07f084ad36b33dae29ffedda5c3cfee081ddd80525eb09569f5f5bad8dc50c/teacher.pt
prepare/cases/447bd127bd902e6d39a48e1501d7d45ccd6ff357cfdba17bd1936a2d4eb1d639/teacher.pt
```

No training implementation, thresholds, configuration or checkpoint was changed
in this review. Any later semantic change to teacher/control preparation needs a
new identity/output namespace; do not overwrite or silently resume the old bank.

## Follow-up: saved-teacher CPU diagnosis (2026-10-27)

User asked whether implementation was wrong and what input is needed. The definite
implementation/acceptance failure is permitting dataset-wide zero motion/reference
support to proceed as the intended comparison. The underlying teacher/qualification
failure is not yet repaired. The .5/.2 cutoffs are inherited engineering heuristics,
not calibrated thresholds established on this Stream checkpoint and street data.

Added `scripts/audit_scene_control_teacher.py`, a local CPU-only reader for existing
trusted `teacher.pt` outputs. No teacher inference, raw video, full checkpoint or new
annotation is needed. It checks a nonempty sibling receipt's size/SHA256, explicitly
marks missing/empty receipts unverified, and rejects mismatched receipts. It reports
full/prefix per-frame score distributions, acceptance/rejection reasons, coordinate
bounds, reference-query agreement and observed-to-future correspondence eligibility.
Optional compressed NPZ exports only tracks/scores/query points for further review.
Nonzero support still requires visual accuracy validation. No threshold is changed
and no new training gate is introduced. Six synthetic CPU tests passed, including
direct CLI use from a different working directory; this is not a real teacher result.

```bash
python scripts/audit_scene_control_teacher.py \
  --teacher-file /path/case000/teacher.pt \
  --teacher-file /path/case004/teacher.pt \
  --output /path/teacher_audit.json --export-npz
```

A bounded read-only attempt to fetch the case000 teacher receipt using the existing
local OBS credential returned HTTP502 on2026-10-27. No teacher payload downloaded.
This does not establish an authentication failure or prove the objects are missing.
Evidence: `D:/workspace/VggAE_DataAudit/scene_control_20261027/diagnose_teacher/fetch_status.json`.
If user can reach OBS Browser, the two paths above plus their sibling `.pt.json`
receipts are sufficient for the first diagnosis; keep them in separate case folders
under D:/中转站. About28–30MB perteacher is expected from the13 dense output frames;
actual object sizes have not been confirmed. Do not request the whole dataset or
5GB checkpoint just to inspect these scores.
