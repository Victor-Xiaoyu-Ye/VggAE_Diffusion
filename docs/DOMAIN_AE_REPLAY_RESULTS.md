# Completed domain AE replay review — 2026-09-14

Inspected the full OBS window_ae_audits inventory and downloaded only small
JSON reports to D:/workspace/VggAE_DataAudit/ae_replay_complete. Two complete
triplets are present, starting09:01 and09:30 UTC. All six audits completed with
launcher exit0 and synced dual-destination publication receipts (no errors).
Receipts report successful publication; the ModelArts mounted destination was
not independently read back. Both attempts have identical per-clip measurements.
They are repeated measurements, not additional independent evaluation samples.

| Cohort | Clips/run | Cached PSNR | Fresh legacy PSNR | Fresh framewise PSNR |
|---|---:|---:|---:|---:|
| single | 32 | 22.41186184 | 22.41185927 | 17.31997317 |
| mixed | 32 | 22.41298079 | 22.41297710 | 17.31966358 |
| historical | 16 | 24.48933113 | 24.48930180 | 18.72770512 |

For every measured clip in all cohorts, cached replay PSNR exactly equals its
previous training baseline. Largest fresh/cache clip PSNR difference is
0.000599 dB in each new cohort and0.000925 dB in the historical control.
Largest future latent relative L2 is0.000208881 across all cohorts; largest
anchor relative L2 is0.000213502. Maximum fresh/cache RGB L1 on0–1 scale is
0.00015741 for new cohorts and0.00015951 for historical. Maximum per-frame PSNR
difference is0.002827 dB for new cohorts and0.006184 dB for historical.
New-cohort discrepancies are within the historical control's numerical scale.
Saved uint8 RGB and fp16 latent rounding preclude requiring bitwise equality.

Replaying/preview time after loading is approximately301–304 seconds per new
cohort and198–202 seconds for historical; this is diagnostic time, not a
diffusion-training throughput benchmark.

## Conclusions and next decision

The observed new-domain AE reconstruction level is real on this fixed street
evaluation set. It is not explained by the previously fixed norm mismatch or
an inability to replay these cached evaluation samples. The historical AE
still reproduces its historical quality. This does not validate every training
cache sample, the entire HQ dataset, or future generated-video quality.

Close the default compatibility investigation; do not rerun stage45 by default.
Accept22.412 dB as the measured frozen-AE reference for the domain-concentration
experiment, with visible detail loss explicitly acknowledged. It is not a
universal quality ceiling or evidence that22dB is sufficient for a final product.
Recommend replacing the inherited old-cohort absolute gate for these two arms
with a reviewed cohort-specific reconstruction/replay policy before fresh
diffusion training, retaining old defaults and strict identity checks. No
threshold was changed or training launched by this review. Preserve existing
v2 caches; restarting diffusion requires a fresh output namespace and RESUME=0.

Single/mixed diffusion has not trained yet. The experiment should now test
whether concentration improves heldout high-noise predictions and generated
structure under matched compute. Known memory64 late-trajectory degradation is
a separate unresolved generator issue; correct AE replay does not solve it.
