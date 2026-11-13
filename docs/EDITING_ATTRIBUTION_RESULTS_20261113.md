# Stage70 complete result review (2026-11-13)

Export: `D:/中转站/VGGAE/scale/editing_attribution_v1`.
All 32 fixed sources x14 arms x seed42 complete: 448/448 evaluation records.
RAE/VAE preparation each32/32; all1248 feedback events applied without fallback.
All2592 receipt-declared media files pass SHA/size; all448 final NPZ load as
17-frame uint8 videos. Local export has no empty files. Shared input identities
and the five common-generator arms' identical SFT checkpoint were checked.
The smoke writer repair is now exercised by a complete production result.

Paper evidence and full audit:
`D:/workspace/VggAE_Paper/notes/STAGE70_RESULTS_20261113.md`.
Reproduction script: `D:/workspace/VggAE_Paper/scripts/import_stage70_evidence.py`.
It reproduces means and source-group bootstrap CIs from recorded per-case metrics;
it does not rerun LPIPS. Runtime code was not changed by this review.

## Scientific outcome

The common-generator group uses one reference-SFT checkpoint without a bridge.
RAE LPIPS .436730 versus RGB .443932, smoothed RGB .441117 and native video VAE
.443928: reductions1.62%,0.99%,1.62%, with32/32 lower against each. Keep improves
against each with bootstrap intervals below0; Track is not better than smoothed
RGB. This rules out separate bridge training as the sole explanation of the
tested readout benefit. It does not establish VGGT-only advantage or superiority
over all VAEs: capacity, temporal lattice, initial-fit objective and pretraining
differ. Gaussian sigma2pixels is not a bandwidth match. VAE is full17-frame,
five temporal slots; RAE uses nine observations with source-flow expansion.

Initial-fitting x online-solving yields LPIPS:
on/on .436943, on/off .436758, off/on .436499, off/off .434379.
Fixed calibrated bridge controls preserve this ordering. Full instead improves
Track slightly; magnitude matching narrows the perceptual gap. Additional
initial calibration and online optimization are therefore not necessary for the
observed perceptual improvement. All arms retain decoded-effect execution.

Main123-source results remain unchanged; new32-source operators are not pooled
into them. Manuscript adds the shared-generator table and complete attribution
supplement, narrows claims accordingly, and retains the user's edited figures.
No new subjective ratings or strong visual demo is inferred from LPIPS alone.

## Remaining three-node work

Highest priority: unused public appearance-edit sources, common legal anchors,
Native/referenceSFT/RGB/DCET, no new training. First audit the staged benchmark's
task/input/target semantics and exclude Stage60 exposed benchmark dev examples.
Current benchmark_records selects V2V/MV2V inputs; this alone does not establish
paired appearance-edit ground truth. Fix selection before generation. Preserve
dual-root I/O, intermediate results, successful-output DI_throughput and resume.
Use independent edit/locality metrics only with valid annotations/evaluator.
Custom17-frame results must not be labeled official benchmark/SOTA.

Fallback: extend frozen operator settings to the remaining91 existing sources,
455 new outputs. This gives wider coverage, not external generalization.
No next-stage launcher or new cluster job is implemented by this review. Do not
repeat Stage70all or prioritize more solver tuning, AE/DiT retraining or long
video. Human evaluation remains deferred at the user's request.
