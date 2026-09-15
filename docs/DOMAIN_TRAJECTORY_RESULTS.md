# Final frozen-domain trajectory review — 2026-09-15

Run r7_domain_single_trajectory_s6000_v1 completed:1792 metrics,
32heldout×2seeds×28arms,48 completed rank statuses, six exits0 and six synced
publication receipts. previews_complete.json=true and16 PNG grids present in
OBS. summary.json's previews_complete=false is the earlier pre-preview marker;
the separate completion receipt records the later finished stage. AE gate
22.41186184 passes with zero reference deviation. Original EMA6000 one-call and
final aggregate metrics reproduce the training evaluation.

| Clean prediction | u | Latent MSE | RGB L1 vs AE | RGB L1 vs RAW |
|---|---:|---:|---:|---:|
|one call|1|.343762|.112238|.121820|
|step8|.875|.372152|.108925|.119202|
|step16|.75|.406121|.109468|.119672|
|step32|.5|.477019|.112780|.122570|
|step48|.25|.543262|.115798|.125261|
|step56|.125|.572051|.116903|.126235|
|final|0|.587732|.117256|.126551|

Final RGB L1 vsAE is worse than one-call for48/64 pairs, worse than step8 for
58/64. Step8 has lowest mean decoded L1 among sampled clean predictions, but
only6.18% lower RAW L1 than final. There is no sharp low-error intermediate
regime resembling the earlier memory64 case. Do not confuse noisy state
decodes with clean x0 predictions or interpret u as video frame time.

The true-path probe at u.015625 has MSE.00031319 and RGB L1 vsAE.00226663
(vsRAW.04648308), while the actual path's x0 has MSE.587732/L1.117256.
The probe contains98.4375% true target in its input, so good local denoising
does not demonstrate video generation. It establishes that near-target
denoising works; it does not prove a numerical solver bug or causal mechanism.

## Visual evidence and decision

Downloaded only three grid images (clips0/3/7,seed101), made lossless crop panels
at frames4/8, and inspected five of those panels. Compared RAW, AE, one-call,
x0 at.5/.25 and final. One-call contains repetitive/rippled textures, averaged
structure and weak details; later predictions introduce visible features but
retain warped building boundaries, doubled trees and smeared surfaces. No
inspected clean intermediate is a satisfactory replacement for final output.
This is limited visual coverage, not all32 videos or both seeds. The grid
design does not include step8/.875, so its modest metric advantage is not a
directly inspected visual-quality result. No further run is required by this
review just to chase that scalar minimum.

The lower one-call MSE did not reveal a good hidden video. It is consistent
with averaging/smoothing; that is an interpretation, not a proven explanation
of training dynamics. Sampling-path errors remain real, but rescuing a good
video solely through early stopping is not supported by this inspection.

Recommendation: close the current R7-window recipe as an unsuccessful quality
baseline under the tested data/model/compute. Preserve checkpoints/reports;
stop default aux/shift/step-count/early-stop sweeps and discuss substantive
representation/generator redesign. This is not proof that all VGGT-based
video generation or all from-scratch RAE diffusion is impossible. Do not claim
the exact root cause (capacity, objective, latent geometry, data) is uniquely
identified. Mixed training remains unrun, so no causal domain comparison.

Local audit: D:/workspace/VggAE_DataAudit/domain_trajectory_review contains small
reports, analysis and comparison panels. Three original30–36MB grids were
removed after crop inspection; originals remain in OBS. No checkpoint or
latent-snapshot downloads, and no new training launched.
