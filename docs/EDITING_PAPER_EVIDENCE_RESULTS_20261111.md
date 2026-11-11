# Stage69 measured evidence and paper update

Reviewed local export: `D:/中转站/VGGAE/scale/editing_paper_evidence_v1`.
The experiment remains the PRE-REPAIR run, identity
`32cac1a143512606c9847fc93fd83888bd31d4f0268e90ac187dd77da3555ad5`.
SFT finished 1,500 updates with no skipped update; configured convergence was
not reached. Of 1,277 planned renders on 123 sources, 1,266 are completed.
Eleven missing outputs on two sources follow the strict confidence-bound
failure described in EDITING_PAPER_RECOVERY_20261111.md. Do not call the
confidence repair completed or restart all training on the strength of this review.

## Paired quantitative result

The original summary has unequal completed row counts. Recompute each table
on its complete source intersection: main121, ablation31, two-seed31. Seeds
are averaged within source before bootstrap; main uses seed42 only. The
main keep-region metric has120 valid sources, not121 (one empty keep region).

| Arm | PSNR | SSIM | LPIPS | Keep L1 | Track L1 |
|---|---:|---:|---:|---:|---:|
| Native VACE |14.87058|.580049|.478027|.121200|.016244|
| Native + propagated reference |13.68946|.505188|.551590|.146230|.018785|
| Matched reference SFT |14.66658|.610377|.461369|.129665|.014820|
| RGB conditioning only |14.71832|.608394|.458113|.128524|.014819|
| RAE conditioning only |14.69300|.609520|.460144|.129403|.014632|
| RGB full |14.73752|.611538|.456232|.128442|.014376|
| RAE full |14.78961|.621647|.450219|.127957|.014045|

RAE−RGB LPIPS=-.006013 (−1.318%),106/121 lower;10000source-paired bootstrap
95%CI[-.007283,-.004781]. Relative reductions versus Native/SFT are5.817%/2.417%.
RAE−RGB Track=-.0003309 (−2.302%),91/121 lower. Keep−RGB=-.0004848 with
CI[-.001191,.000152], so no reliable keep advantage; Keep is worse than Native.
These metrics describe agreement with dataset-produced edited targets and
independent Farneback transport, not human edit execution or3D geometry.

The representation-by-feedback interaction is the useful new insight:
`(RAEfull−RAEbase)−(RGBfull−RGBbase)=-.008044 LPIPS`,120/121 larger RAE gains.
Reproducible sorted-source bootstrap seed42 gives95%CI[-.008837,-.007278].
RAE conditioning alone is slightly worse than RGB, but effect feedback reverses
the ordering. This supports the present decoded-state execution interface;
it does not prove a uniquely VGGT geometry advantage or generic decoder necessity.

## Mechanism and cost

On31 paired sources, LPIPS is .448397 for RAEbase, .447538 bridge-only,
.438190 velocity-only, .438185 full. Initial bridge conditioning remains in
these controls. The online velocity path carries the main observed improvement.

`rae_reinject` gives .438028 LPIPS versus .438185 full; paired full−reuse
.000157 CI[-.000216,.000485]. It RETAINS the initial four-step decoder fit in
`prepare_rae_state`, effect decoding, shared proposal and velocity feedback.
It is not a no-decoder/no-calibration control. Matched31 sampling188.261s full
versus94.368s reuse (49.9% lower); logical RAE calls328.29 versus54. The
initial state prep and model loading are excluded. Full slightly improves
Track (.012938 versus .012963) but extra online optimization is not proven
necessary for perceptual improvement.

No-temporal/no-graph LPIPS~.4395, shuffled online pulls .429615, unit confidence
.435354. These counter-controls do not support correctness/necessity claims
for current online geometry or reliability. All retain the original source-
correspondence-derived cached RGB proposal. Do not remove these outcomes while
claiming those modules are established contributors.

Two-seed31-source meanLPIPS: Native.463406 / RGB.442197 / RAE.435385.
Main121 samplingmean: Native65.779s / RGB182.549s / RAE188.920s;
peakallocated36.602 /37.026 /40.009GiB. Timing includes online feedback,
decode and intermediate publication, not initial preparation/loading.

## Media, manuscript and next action

The local export still contains265empty files:161JSON,51MP4,52NPZ,1log.
The user will re-download them. Complete phase metrics match all1266 successful
generation IDs and shared input identities. A media audit decoded2145 present
MP4 and1212present NPZ; many intermediates/.pt files were not downloaded.
Do not equate local export gaps with a fresh OBS or cluster sync failure.
All121 common-source overview sheets were reviewed; selected cases were also
reviewed at5 times. Benefits over RGB are subtle; shared wrong/overbroad color
changes remain. No blind human scores have been collected.

Paper repository `D:/workspace/VggAE_Paper` now uses the121paired main table,
an interaction plot, writeback/state-reuse controls, two-seed/complete-control
appendix, and actual matched brush/fabric examples. Redundant old debugging
tables/failure montage are omitted from the narrative; key controls and scope
remain. Initial RAE training and its measured recipe stay explicit.
Paper scripts archive measurements/checksums and reproduce all189mean/count
pairs, which agree with the independent review to1e-12. No runtime/training
code was changed by this analysis. Main manuscript stays8pages before refs.

Next: existing `PAPER_STAGE=repair` on three nodes can recover only11missing
outputs without retraining. Until repair evidence arrives, keep coverage123
planned/121paired and human packet93/96, rather than asserting suite completion.
