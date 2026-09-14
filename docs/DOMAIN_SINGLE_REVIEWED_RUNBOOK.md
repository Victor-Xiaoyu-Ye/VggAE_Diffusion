# First reviewed single-domain diffusion run

User approved proceeding after complete AE replay. Stage46 defaults to single
street cohort, fresh namespace r7_domain_single_uniform_reviewed_v1. It reuses
the validated v2 train8192/eval128/other128 caches and frozen2048-video cohort;
no AE or dataset recaching. Stage43 still verifies CSV hashes and cache counts.

Update ascend-910b and use the existing ModelArts dependency initialization.
All six nodes (8 NPUs each) run:

```bash
bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/46_train_domain_single_reviewed.sh
```

Training: 6000 steps, global batch96, width768/depth12, x0/uniform, learning
rate1e-4, warmup300, no auxiliary loss, first-frame condition without text.
Every500 steps: checkpoints and32 heldout clips ×2 seeds ×online/EMA evaluation,
with8 preview clips. Throughput every10 steps; incremental output publication
every60 seconds plus final publication. Existing stage29 full resume and dual
source checkpoint staging are retained. This is a quality experiment, not a
promise of good videos. The unresolved memory64 late-sampling degradation is
not repaired by changing the data domain.

Only this entry uses a reviewed AE policy: minimum mean22.3dB plus exact32
video IDs, AE signatures, legacy norm, finite metrics and per-video absolute
PSNR deviation <=0.05dB against configs/ae_reference_single_v1.json. The measured
mean is22.41186184dB. The0.05dB tolerance is an explicit engineering margin,
not an empirically derived quality cutoff; measured fresh/cache deviations
were <0.001dB. A drift in either direction beyond tolerance stops training.
Other launchers retain23.5dB default unless explicitly overridden as before.

The reference content SHA256 is part of the resume identity, independent of
its local path. A copy is saved as reviewed_ae_reference.json. ae_baseline.json
records reference_check, measured deviations and gate_passed. Old checkpoint
contracts remain unchanged when no reference is supplied. No checkpoint from
the failed step0 v2 jobs is resumed. Never point this entry at old output roots.

Outputs, in current OUTPUT_URL and persistent owner output:

```
scale/r7_domain_single_uniform_reviewed_v1/
```

Review at500 and1000 steps, then the full6000 trajectory. Compare generated
structure/motion, high-noise prediction and RAW/AE errors, not just train loss.
The fixed heldout evaluation is a development set, not a final unbiased test.

For continuation of this exact run only, with unchanged48-card topology:

```bash
RESUME=1 bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/46_train_domain_single_reviewed.sh
```

Do not rerun the fresh command against an existing namespace; the output guard
will reject it. New independent repetitions need a fresh WINDOW_NAMESPACE.
This task prepares and validates the launcher locally; actual NPU training is
started by the user's ModelArts job, not by the local Windows environment.
