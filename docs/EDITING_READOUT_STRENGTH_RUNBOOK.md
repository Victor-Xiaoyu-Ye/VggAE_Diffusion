# Stage72: shared-generator readout strength and coverage

Implemented 2026-11-14. One coordinated **3-node / 24-NPU** job. This is an
inference/evaluation experiment using the trained C256 codec and reference-SFT,
not another AE, DiT, LoRA, anchor, or state-bridge training run.

## Why run this

Stage70 already compares the five operators under the SAME reference-SFT, with
no learned state bridge: SFT without feedback, RGB, spatial Gaussian RGB,
full-video native VAE, and trained C256 RAE. Its 32-source results favor RAE on
LPIPS, but equal feedback alpha does not establish equal visible editing.
Stage71 also demonstrates that similarity alone can reward insufficient editing.

This job addresses two remaining questions:

1. Does the RAE/control tradeoff persist at comparable observed editing strength?
2. Does the fixed .35 operator comparison extend from the original 32 sources to
   the other 91, using the complete pre-existing 123-source cohort?

It does not retune the 20 public colorization cases, establish VGGT-specific
geometry necessity, or replace a same-protocol external-editor comparison.

## Exact matrix and order

| Phase | Sources | Outputs | New generation |
|---|---:|---:|---:|
| P0 original operators at .35 | Existing 32 | 160 | Normally 0; verified imports |
| P0 four feedback operators at .15 and .55 | Same 32 | 256 | 256 |
| P1 five original operators at .35 | Remaining 91 | 455 | 455 |
| Total, unique outputs | Original 123 | 871 | Normally 711 |

The no-feedback SFT output is shared across the strength comparison; there are
no meaningless SFT alpha variants. Original parent record order and subset IDs
are frozen before generation. No outcomes enter selection or seed assignment.
Keep 17 frames, 8 fps, 30 sampling steps, seed base 42, Stage70 scheduler/CFG,
three prescribed feedback sigmas, .25 per-channel/time native RMS cap, original
hard anchor, initial four-step RAE/native-VAE readouts and spatial sigma2 RGB.
All four feedback operators use the same gate and native encoding-difference
writeback. No online solver, state bridge, or native-context synchronization.

First complete and publish **summary_subset.json** for P0. Then prepare the
remaining 91 readouts, render their five arms, and publish the complete report.
RAE and native-VAE preparation run separately before loading the 14B generator.

## Launch on all three nodes

Use the normal existing bootstrap that stages this repo and vggae_ref. No new
entry point outside scripts/scale and no new model/data downloads are required.
All three nodes run the identical command; ModelArts supplies node rank/hosts.

```bash
cd /cache/yexiaoyu/VggAE_Diffusion
READOUT_STAGE=smoke bash scripts/scale/72_audit_edit_readout_strength.sh &&
READOUT_STAGE=all bash scripts/scale/72_audit_edit_readout_strength.sh
```

Smoke uses three original plus three extension sources, distributed over all
three nodes, six sampling steps and **54 new outputs**. It retains the real
four-step readout fitting. Existing compatible prepared effects may be reused;
full-run generated videos are never substituted for smoke sampling. Separate
namespace `editing_readout_strength_v1_smoke`; full is
`editing_readout_strength_v1`. Do not override the frozen alpha grid after
looking at these test outcomes.

Default parents:

- Stage69: `editing_paper_evidence_v1` (the repaired export called v2 locally
  retains this original runtime identity).
- Stage70: `editing_attribution_v1`.
- Evaluation bank: `vace_edit_pilot_v1`.
- Native assets: `/cache/yexiaoyu/vggae_ref/Wan2.1-VACE-14B`.
- Frozen trained Scene C256 AE and StreamVGGT: existing staging helper/paths.

Default coordinator port base is 31600, join timeout 7200s. Budget remains 3x8;
the driver never implicitly allocates 8 nodes. Based on previous operator
sampling (86s SFT, about101s feedback per output, about36.85GiB peak allocation),
pure generation is roughly one hour when fully occupied. Allow **about3–6h for
the full pipeline after assets are present**, including repeated model loads,
prepared-effect recovery, 91 new fits, CPU LPIPS/flow metrics and dual media
readback. This is a planning estimate, not a measured Stage72 runtime. Bootstrap
download delays are additional. Cached imports do not contribute to DI_throughput.

## Strength measurements and interpretation

- Every new feedback event reports actual post-cap clean latent RMS, future-only
  latent RMS, velocity-change RMS, cap fraction and elapsed time. Reference
  prefix and first video slot are excluded from the future-only quantity.
- Old Stage70 events retain their original generation identity/fields. Do not
  fabricate the newly added future-latent diagnostic for historical events.
- Every final output, including imports, measures future-frame source changes
  on the SAME source/reference support: RMS/L1, outside-support RMS, and signed
  projection onto the shared reference's edit direction. Frame0 and letterbox
  padding are excluded. These are observed editing/reference-agreement proxies,
  not human judgments or instruction-success labels.
- Evaluation retains LPIPS, edit-region target error, background preservation,
  independent source-flow tracked-effect error and frame-delta/motion diagnostics.
  Dataset future targets are opened only in the evaluation phase.
- Report all three fixed alpha points and paired differences. No “best alpha”
  is selected. For an additional diagnostic, interpolate only within each
  source's common measured-strength interval, at its midpoint. Reject flat,
  nonmonotonic, missing and non-overlapping curves and expose their counts.
  Use both edit RMS and reference-direction projection as strength axes.
  **Interpolated values are not newly generated outputs or a causal proof.**

Small/no overlap is an unresolved comparison, not a reason to remove sources
from the fixed-alpha tables. Scientific outcomes never gate the extension job.
If gains disappear at comparable editing, narrow the claim accordingly. If
they persist and the new91 agree, this supports practical readout advantages
under the shared generator, still not equal codec capacity or temporal lattice.

## Persistence, resume and failure behavior

Writes go to the usual `$OUTPUT_URL/scale/editing_readout_strength_v1` and the
persistent owner OBS `output/scale/editing_readout_strength_v1`. Both are read
for recovery and verified/reconciled for media. Prior remote roots are read-only.
Imports retain original payload SHA256, generation identity and origin receipts;
the new suite uses a separate job identity. Imported preview media are copied
with new receipts, while original state snapshots remain in Stage70 and are
explicitly linked. New intermediates save tensors and actual correction videos.

```bash
# Same full namespace: recover completed work and retry only missing/failed work.
READOUT_STAGE=resume bash scripts/scale/72_audit_edit_readout_strength.sh

# Evaluation/publication only, if all outputs are already complete.
READOUT_STAGE=evaluate bash scripts/scale/72_audit_edit_readout_strength.sh

# Smoke resume on a fresh node also restores its durable configuration.
READOUT_NAMESPACE=editing_readout_strength_v1_smoke READOUT_STAGE=resume \
  bash scripts/scale/72_audit_edit_readout_strength.sh
```

Independent failed arms do not prevent the other operators or the extension
from running. Nonzero exit and exact coverage/failures are retained; successful
files are not discarded. Source/model identity mismatches stop before expensive
work rather than mixing different experiments. Missing/corrupt files are never
reported as quality failures or silently treated as completed observations.
The existing idle guard spans preparation, waiting, CPU evaluation and summary;
only actual new computation is included in DI_throughput. CPU summary/evaluation
disable Ascend backend autoload. Cleanup remains bounded.

Key artifacts:

- `manifest.json`, `plan.json`, per-node launcher logs/coordinator status.
- `calibration/`, `vae/`: imported provenance or fitted effects/states.
- `render/<arm>/seed42/<id>/`: result, lossless17frame NPZ, comparison video,
  metrics, sealed receipt; new `feedback/step_*.pt/.json/.mp4` intermediates.
- `summary_subset.json`: P0 report published before P1 starts.
- `summary.json`: 123-source, old32 and new91 tables; fixed-alpha paired tables,
  strength-overlap diagnostics, costs, coverage and failures.
- `evaluations.json`, `media_sync.json`, `phases/`, `progress/` and `logs/`.

## Local verification and remaining cluster checks

The Stage69/70 production manifests, code hashes, operator config, SFT receipt
and exact123/32 source identities were checked against the local completed
exports. **29 unique CPU/Bash tests passed**, along with five-file Python compile,
launcher Bash syntax and whitespace checks. Tests exercise actual small VACE sampling, native full-video VAE fitting,
RAE state-gradient fitting and expansion, actual MP4/NPZ publication, old output
import, changed-generation rejection, cache-excluding throughput, fresh-local
mirror recovery and primary repair, same-output observation, independent-arm
failures, actual launcher ordering, CPU isolation, smoke and durable-config resume.
These use explicit small model/codec fixtures; they do not establish production
910B memory/kernel behavior, HCCL/OBS availability, runtime or quality.

After results, update the paper with complete paired tables/curves and declared
failure coverage. External editors remain a separate task pending ready weights,
offline910B compatibility and a matched input protocol; no unsupported row is
added to the manuscript by this implementation.
