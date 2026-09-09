# Single-variable shift3 training experiment

Entry on all six ModelArts nodes: `bash scripts/scale/33_train_window_shift3.sh`.
Use the usual repository/environment staging,48 Ascend910B cards. This is a
fresh random-initialized6000-step experiment, not a warm start from EMA6000.
Source cache and frozen legacy t2v2 AE are unchanged. Stage29 provides the
existing quality gate, double reads/writes, periodic previews, full training
state checkpoints and resume. Stage30/31 cache rebuilds are not invoked.

Output: r7_window_t2v2_legacy_x0_shift3_v1. Default source cache:
r7_window_t2v2_legacy_diag_v2. Set RESUME=1 to resume THIS new run; the original
shift1 checkpoint fails the immutable flow/args contract and is not accepted.
STOP_AFTER_STEPS supports a planned pause without changing the6000-step LR
schedule. A repeated fresh attempt needs a new identical WINDOW_NAMESPACE on
all nodes; never overwrite earlier results.

## Evidence and hypothesis

Stage32 ruled against a dominant sampler-step problem:128 steps gave only1.21%
heldout RAW L1 improvement. Training previews also fail; first-frame conditioning
works. The next experiment tests whether increased high-noise training exposure
helps generation. This is a hypothesis, not a diagnosed sufficient cause.

Original training samples u=sigmoid(N(0,1)), with u=1 pure noise. x0 objective is
MSE/max(u,.05)^2, which is velocity-equivalent above the clamp. At EMA6000:

| u | heldout clean MSE | corresponding weighted objective |
|---|---|---|
| .05 | .002802 | 1.120740 |
| .25 | .042522 | .680360 |
| .5 | .118113 | .472454 |
| .75 | .208557 | .370768 |
| .95 | .298327 | .330556 |

Low clean-error values are not necessarily small weighted losses. Conversely,
these fixed-time heldout measurements are NOT integrated training loss shares
or gradient attribution. Higher noise is intrinsically harder; these values
alone do not prove a faulty objective. No optimizer/gradient-scale fix is claimed.

Change only time_shift1->3: u'=3u/(1+2u), equivalently sigmoid(N(0,1)+ln3).
P(u>.75) goes13.60%->50%; P(u>.9) goes1.40%->13.60%. The x0 weighting,
loss floor, initialization seed, model178M, learning rate/schedule, batch96,
AE/cache, no-text setting, EMA and Euler64 sampler remain unchanged. This also
changes effective weighted-objective balance; do not claim it isolates exposure
independently from that balance. time_shift affects training draws only; sampling
retains the same uniform descending noise-time grid.

## Measurement additions shared by the trainer

- normalization_audit.json: exact cached per-slot/channel means/stds and ranges.
  Local downloaded logs had no stats.pt, so actual numerical ranges await cluster
  startup; no claim that whitening/covariance has been validated.
- train/noise_bin0..4: [0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1], global sample
  counts/fractions, mean u/weight/clean MSE/objective/target power, loss fraction.
  Aggregated across all ranks and microbatches since last log. Loss fraction
  measures scalar objective contributions, not gradient norms.
- train/normalized_cond and normalized_target: empirical temporal-slot/channel
  means/std ranges across the logging window (batch/spatial axes reduced).
  Short correlated data windows need not have exact0/1 moments.
- eval online/EMA: fixed-noise x0 errors now include pure noise u=1; normalized
  target/generated power; RAW/copy and noise errors are also in aggregate metrics
  and TensorBoard. Existing per-clip metrics/previews are retained.

Telemetry is detached, does not consume RNG or change losses. It adds some
measurement overhead. DI training step timing includes local accumulation but
excludes post-step log reductions and evaluation/save, as previously documented.
On full resume telemetry starts a fresh reporting window; optimizer/EMA/LR/RNG/
data cursor resume semantics are unchanged. No new CLI flags were needed.

## Decision criteria

Compare matched500/1500/3000/6000-step EMA previews and RAW/AE errors against
shift1 v2, using the same16 heldout clips and seeds. Prioritize visible structure,
appearance and motion; a drop in train/loss across different u distributions is
not a quality gain. A valid test should show the intended high-noise bin exposure
and reasonable latent moments. If high-noise denoising improves but free samples
remain distorted, reject it as sufficient rather than automatically extending.
No acceptance claim for reasonable video or geometry follows from this arm alone.

Validation: CPU oracle/telemetry/shift-distribution tests plus existing exact
training-resume and double-read/write tests; real shift3 NPU quality pending.
