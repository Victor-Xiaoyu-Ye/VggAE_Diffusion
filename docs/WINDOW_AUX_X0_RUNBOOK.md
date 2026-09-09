# Intermediate clean-R7 supervision

Stage36 adds a separate depth control: block8 instead of block6, weight0.5
and every other stage33 setting unchanged. Entry36_train_window_aux8_x0.sh,
namespace r7_window_t2v2_legacy_x0_shift3_aux8_w05_v1, fresh6000 steps.
Do not resume aux6 into aux8 or modify a running aux6 job. Same-aux8 full
resume is supported. Aux8 is a hypothesis, not a promoted quality winner.
User will finish aux6 and run this separately, then compare complete curves.
Early aux6 EMA500 RAW L1=.141345 vs baseline .140643, latent MSE=.792394
vs .808685: no clear RGB benefit yet. Same300–500-step DI median5034 vs5104.

Stage35 is a fresh6000-step controlled arm against stage33 shift3. Same
legacy t2v2 AE/cache, noise shift3, data/seed/batch96, optimizer, EMA,
Euler64,16 eval clips x2 seeds and500-step checkpoints/previews.
No AE training or cache rebuild. No warm start from EMA2500/6000.

One intervention: after block6 of12, LayerNorm+linear192-channel clean-target
head. Loss=main weighted x0 loss +0.5*auxiliary weighted x0 loss, both using
the same noisy input/target/time and 1/max(u,.05)^2 weighting. Head starts
zero; intermediate backbone gradients from it begin after the head updates.
Initial baseline parameters/RNG are preserved. Weight0.5 is a fixed initial
hypothesis, not tuned or guaranteed optimal. Shared gradient clipping remains1;
the auxiliary term can change total gradient magnitude and clipping frequency.

Sampling/evaluation use only the final head, no auxiliary guidance or extra
DiT pass. No VGGT pass in training. Auxiliary parameters are149184 at width768,
included in optimizer/EMA/checkpoints. Model and objective contract reject
resuming baseline into auxiliary or changing layer/weight on full resume.
Disabled auxiliary defaults preserve historical baseline resume contracts.

Stage35 wraps33 then29, retaining dual-source inputs, incremental dual outputs,
intermediate artifacts and full optimizer/scheduler/EMA/RNG/data cursor resume.
Set RESUME=1 only for the same auxiliary namespace. Fresh default:
r7_window_t2v2_legacy_x0_shift3_aux6_w05_v1.

Logs include train/main_loss, train/aux_loss (unweighted),
train/aux_weighted_loss and train/loss(total). Existing noise-bin telemetry
continues to describe the MAIN head, not total/auxiliary gradients. Compare
main loss, heldout RGB/latent errors, fixed-noise errors and previews across
the full curve; lower total loss is not the selection rule. Compare DI
throughput/step seconds/peak memory after startup; overhead is unmeasured on
910B and total wall time includes evaluation and publication.

This is borrowed deep supervision, not geometry novelty. It does not yet
implement decoder-aware or geometric relation loss. If quality improves,
geometry supervision must beat this plain-latent auxiliary control separately.
Current perturbation evidence is a12-video partial subset accepted by user
for deciding the next arm; it does not exclude all decoder sensitivity.

Local validation covers auxiliary backward routing, unchanged main inference,
initialization/RNG parity, exact synthetic full-state resume with auxiliary,
disabled-contract compatibility and existing window tests. Real NPU/HCCL/OBS
and quality must be validated by the run. No reasonable-video guarantee.
