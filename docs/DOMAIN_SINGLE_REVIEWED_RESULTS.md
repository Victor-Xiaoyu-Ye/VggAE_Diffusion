# Single-domain diffusion results — 2026-09-15

Run: r7_domain_single_uniform_reviewed_v1. All48 rank receipts finished6000;
six launcher exits0 and six publication receipts synced. The evaluation file
contains1536 unique rows:12 steps ×32 heldout videos ×2 seeds ×online/EMA.
AE reference check passes with zero per-video PSNR difference and22.41186184dB
mean. Remote checkpoint objects exist; no2.86GB checkpoint was downloaded or
internally inspected. Receipt success does not independently verify mounted
ModelArts destination bytes.

| Step | EMA RGB L1 vs RAW | EMA RGB L1 vs AE | EMA latent MSE |
|---|---:|---:|---:|
|500|0.244462|0.239075|1.316279|
|1000|0.163494|0.156560|0.911865|
|2500|0.130568|0.121718|0.628838|
|3000|0.129075|0.120105|0.614279|
|6000|0.126551|0.117256|0.587732|

The best observed EMA RAW metric is at6000. Second-half RAW improvement is
only1.96%. At6000, RGB-copy RAW L1 is0.12658492 versus generation0.12655111:
only0.027% relative improvement. Averaging the two seeds per video,15/32
videos beat copy. AE reconstruction RAW L1 is0.04646563, far below generation;
the decoder's reconstruction loss alone cannot account for the gap. Pixel L1
against one future is not a perceptual/generative quality metric; alternative
valid motion can increase it, so the copy comparison is not a stand-alone test.

Visual coverage: inspected final EMA seed101 clips0,3,7 at frames4 and8, plus
clip0 frame4 at500 and2500. Streets/layout are recognizable at6000, unlike500;
facade edges warp/double, distant details smear and foliage/road textures drag.
These inspected frames do not meet the intended quality baseline. This is a
limited preview audit, not a full subjective evaluation of all32 videos.
Six small MP4s (clips0/3/7,seeds101/211) are available locally under
D:/workspace/VggAE_DataAudit/single_reviewed_results/samples/step0006000/.

Final generated adjacent-frame L1 motion0.067045, AE0.063039, RAW0.081061.
Nonzero/matching magnitude is not proof of correct camera motion or geometry.
Median training DI throughput from loggedsteps1000–6000 is5213.77 and median
step time0.4971s; includes neither whole-job staging nor evaluation/publication
cost in that per-step timing.48×batch1×accum2,6000steps =576000 window exposures
over8192 cached windows (about70.3 exposures/window).

## Implications

AE/cache compatibility remains validated on the fixed eval set. Training is
stable and learns recognizable structure, but domain restriction alone has
not produced the desired quality in this run. Mixed comparison has not trained;
older experiments use different eval cohorts and cannot establish a causal
single-vs-mixed advantage from aggregate scores.

For the same final EMA eval noise draws, one-call pure-noise x0 MSE0.343762 is
lower than final sampled MSE0.587732 (about1.71×). This motivates decoding the
one-call predictions and auditing current-checkpoint sampling trajectories.
It does not by itself prove a solver bug: conditional averages can have lower
MSE than diverse samples. At present there is no decoded one-call visual or
intermediate trajectory from this new checkpoint. Earlier memory64 trajectories
do provide independent evidence of late degradation, but cannot identify where
the new model changes without measuring it.

Retain final6000/EMA as this run's reference; use checkpoint_final.pt with
internal step validation for a follow-up, rather than assuming the contents of
the best-named object from its filename. Prioritize frozen same-noise one-call /
intermediate / final RGB and train-heldout comparisons before extending steps
or another auxiliary/shift sweep. No new training or diagnostic launcher was
implemented by this review.
