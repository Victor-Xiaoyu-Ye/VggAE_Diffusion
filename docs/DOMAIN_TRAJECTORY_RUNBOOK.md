# Frozen single-domain EMA6000 trajectory

Stage47 tests the remaining hypothesis before redesign: can this checkpoint
produce reasonable decoded one-call/intermediate predictions that final
sampling damages? It does not train, change the solver or load future RGB as
conditioning. GT probes are oracle diagnostic branches, never fed into sampling.

After updating ascend-910b and the usual ModelArts dependency initialization,
all six nodes execute:

```bash
bash /cache/yexiaoyu/VggAE_Diffusion/scripts/scale/47_diagnose_domain_trajectory.sh
```

Read-only checkpoint_final.pt from r7_domain_single_uniform_reviewed_v1, internal
step6000 required, EMA only. Uses all32 original eval clips and original seeds
101/211, with noise index0–31 (no memory64 offset). The checkpoint manifest,
statistics, AE signature and reviewed reference hash must match. Per-clip AE
replay and checkpoint meanminimum22.3 remain checked; no old23.5 override trap.
No train windows are selected: this is exactly the failed heldout baseline.

Uses production WindowFlow Euler64 observer at indices0,8,16,32,40,48,56,60,63,
u=1,.875,.75,.5,.375,.25,.125,.0625,.015625, then final. Each node saves state,
predicted x0 and true-path probe, then final;32×2×28=1792 metric rows expected.
At u1, x0 is the one-call prediction. State decodes contain noise and are not
clean video candidates. Probe decodes use the true target and are not generation.

All metric rows and CPU latent snapshots are persisted before previews.
First8 clips ×2 seeds get PNG grids containing AE, RAW, one_call, x0 at.5/.375/
.25/.125/.0625/.015625 and final. No MP4 or separate frames are produced in this
diagnostic.48-rank status, per-stage progress, diagnostic DI throughput,
dual-source checkpoint staging and incremental/final dual publication reuse
stage29. This is a real distributed run, not the stage45 serial coordinator.

Output under current OUTPUT_URL and persistent owner output:

```
scale/r7_domain_single_trajectory_s6000_v1/
```

Check metrics.jsonl/summary.json (1792 unique rows), previews_complete.json,
all48 diagnostic status files and six launcher/publication receipts. Large
latent/checkpoint artifacts stay in cluster/OBS; local review should fetch only
small reports and selected previews to D. Use a new WINDOW_NAMESPACE on rerun.

Decision: if actual one-call/intermediate clean predictions have reasonable
structure and degrade later, focus on trajectory correction. If every candidate
is structurally poor, do not infer that a smaller MSE rescued generation;
prioritize representation/generator redesign rather than another sampler sweep.
This is evidence for deciding, not a guarantee either path will succeed.
CPU tests check observer agreement and zero-offset EMA-only seed alignment;
new NPU diagnosis and visual outcomes remain pending.
