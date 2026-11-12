# DCET: closing the evidence chain on three nodes

Date: 2026-11-12 (Asia/Shanghai). Budget: three nodes, eight Ascend 910B devices
per node. This is a prioritized experiment specification, not a claim that new
launchers or cluster runs have been completed. The existing Stage69 repair is
runnable; groups A/B/C below still need implementation and smoke testing.

## 1. What the paper can currently support

The defensible core is: a reconstruction-trained, decodable scene state can be
used as an executable edit interface for a pretrained video editor. Its visible
effect, rather than its feature-conditioning value alone, improves the measured
paired-target output under the current protocol.

| Contribution | Existing evidence | Remaining gap |
|---|---|---|
| Train a usable hybrid scene codec | C256 reconstruction and recipe records | Not a new RAE paradigm; training tricks are not separately attributed |
| Calibrate state-dependent visible effects | Stage62 H1: 24 development sources, synthetic known transforms, edit-region error reduced 64.09% vs direct token warp | This decoder-defined, fully observed target is a mechanism probe, not real-video editing superiority |
| Execute the state inside native denoising | Stage69 bridge/velocity interventions; velocity-only nearly matches full | Initial trained bridge remains; do not claim it can be removed |
| RAE and feedback interact | Common121 LPIPS: Native .478027, SFT .461369, RGBfull .456232, RAEfull .450219; interaction -.008044 | Does not identify initial calibration, VAE alternatives, or VGGT-specific geometry |
| Efficient calibrated-state execution | Same31: reuse94.4s vs full188.3s, similar LPIPS | Reuse retains initial four-step fitting; not a no-calibration result |

Core system claims mostly correspond to results. The paper's stronger mechanistic
reading is not yet closed: decoder calibration must be connected to final output,
and the representation/control alternatives must be separated. Better LPIPS is
not itself proof of better instruction following or perceptible user benefit.

## 2. Priority queue

### P0: recover existing evidence and start human review

Run the existing repair on the same three nodes and original namespace:

```bash
cd /cache/yexiaoyu/VggAE_Diffusion
PAPER_STAGE=repair bash scripts/scale/69_complete_editing_paper.sh
```

Recover the11 missing renders on2 sources, regenerate paired123/subset32 tables
only after completion, and reconcile both output roots. Do not restart SFT or
replace successful outputs. Local empty-file redownloads are a separate issue.

The existing blinded packet compares RAEfull to Native/RGBfull/RAEbase:32x3=96
trials after repair (currently93). Add SFT as a fourth comparison using already
generated videos, giving128 pairs, not new GPU inference. Target five independent
raters:640 paired judgments, each with instruction, preservation, temporal and
overall axes. Randomize A/B and trial order, retain ties/neither, summarize by
source and report counts. A prepared packet is not completed ratings. This needs
human availability, not training cards. The SFT packet extension is not implemented.

### P1-A: initial calibration must affect the final video

Use the fixed32 source IDs from Stage69 (31 until repair), identical anchors,
proposal, selected RAE checkpoint, seeds42/43, solver schedule and write strength.
This is a deployment intervention, not a retrained architecture comparison.

Four cells: initial decoder calibration on/off x online optimization on/off.
Off initially means the encoded shared proposal before the four fitting steps;
it must not reuse the already calibrated cache. Off online retains actual effect
decoding and velocity injection. Reuse compatible existing cells by hash.

Add a no-calibration magnitude control: match the calibrated correction's RMS
within the source-derived edit support, computed without any future target, to
distinguish response quality from a simple smaller/greater edit dose. Document
that this diagnostic uses the calibrated arm's magnitude, not hidden target RGB.
Record actual dose, saturation, latent norms, and final LPIPS/Edit/Keep/Track.
Do not optimize or select strength against the evaluated targets.

Read out both calibration effect error against the shared proposal and final
video metrics. A lower fitting loss alone does not establish a contribution.
Separate bridge-state changes from feedback in a fixed-initial-bridge check on
the same sources if the whole-state intervention is ambiguous. No new LoRA/AE
training is needed for these deployment tests.

Use the old Stage62 H1 as the bounded motivation; do not spend another run on
the same synthetic affine probe. If initial calibration has no reliable final
benefit, keep decoded-effect execution as the mechanism and narrow the name/claim
instead of asserting that repeated online optimization is essential.

### P1-B: compare writable spaces with the same generator

Use the same no-state reference-SFT checkpoint for ALL of these controls:
no feedback, RGB effect feedback, bandwidth-matched smoothed-RGB feedback,
native Wan video-VAE effect feedback, C256-RAE effect feedback. Set the simple
RGB smoothing bandwidth from development response spectra, with the same support
and correction cap, to test whether generic low-pass regularization explains the
gain. Do not insert VAE tokens into a bridge trained for C256 or change
the native LoRA together with the representation. The learned state bridge is
absent in this operator comparison; the existing full DCET remains a separate
system row. A positive result here is not a complete separately trained VAE-state
system comparison, and cannot establish a VGGT-only causal advantage.

All arms receive the same legal anchor, source-derived proposal, support and
correspondences. They use the same three feedback events, native steps/CFG,
encoding-difference writeback and correction cap. Implement decoder residual
readout around the SOURCE state for both codecs. The VAE sees the full17-frame
window with its native temporal compression and correct scale; do not repeat the
Stage62 T=1 proxy and label it a video-VAE comparison. Report codec sizes/context,
reconstruction, optimization calls, actual edit dose and elapsed time.

Fit step size and numerical scales only on the old24 development examples with
equal tuning budgets per space. Freeze choices before evaluating the32 sources.
Report both equal feedback-event budget and a measured time-budget comparison;
equal iterations do not imply equal cost. One matched-time reference-SFT sampler
and RGB correction control on the same32 are more informative than the already
run, unmatched60-step native-reference row. Include all preprocessing/calibration
and sampling time separately. Do not silently choose the most favorable budget.

No new foundation weights are needed for the VAE control. Full-video VAE gradient
memory is UNMEASURED; verify numerical gradients, temporal alignment and peak NPU
memory first. Sequentially place codecs or offload as needed, preserving exact
temporal semantics. Re-encode the same corrected RGB in the native VAE; do not
bypass the common writeback equation just because the control uses native latents.

### P1-C: real editing benefit and a second source

Human review runs alongside A/B using existing outputs. Separately use up to64
public source clips from the already staged VACE-Benchmark, after checking its
actual tasks/metadata. Fix source IDs and appearance-edit instructions before
generation; exclude unsupported topology/action edits by task definition, not
by generated success. Deduplicate against the full training/dev exposure list.
If fewer compatible clips exist, report the actual count; do not fill from
unrelated VACE tasks or call a custom subset the official leaderboard.

Generate Native, matched SFT, matched RGB feedback and fixed full DCET under the
same17-frame protocol and shared anchor. Source motion/visibility strata can be
fixed before generation; they do not certify 3D geometry. Reuse text/CLIP assets
only after checking the exact evaluator and offline weights. Include edit-direction
agreement, independently defined locality, natural motion and blinded quality;
image-text scores remain proxies. A source-target difference mask alone does not
identify semantic edited objects. Manually annotate a small32-source evaluation
subset if needed; these masks never become model inputs.

Add one compatible external editor, preferably AnyV2V with exactly the same edited
first frame, if its weights and Ascend runtime are ready. Timebox integration to
four engineering hours before reconsidering dependencies. Preserve its supported
aspect ratio/window and report any common-time evaluation explicitly. Do not
present a shortened/resized custom evaluation as reproduced published scores.
The four existing-system arms should not wait for this external integration;
external comparison remains an open evidence item if it cannot be run.

Checked primary sources on2026-11-12:
- VACE task/benchmark input contracts: https://github.com/ali-vilab/VACE
- AnyV2V accepts a shared edited first frame: https://github.com/TIGER-AI-Lab/AnyV2V
- EditDistill weights table still says coming soon, with a placeholder download
  identifier: https://github.com/yu-shaonian/EditDistill . Do not block this queue
  on unverified weight availability or substitute its published table for a run.

## 3. Three-node schedule

Use ONE3x8 job group at a time, sharded by source. Do not run three copies of the
same global coordinator or assign different stages to workers in one process
group. Human review and local paper analysis can run concurrently without NPU.

| Order | Reserved elapsed slot | Three-node work | Deliverable |
|---|---|---|---|
| 0 | 0-4h | Stage69 missing-only repair, metric/dual-root reconciliation | Complete cohort and blinded packet |
| 1 | next6-10h | Implement/test A, then32-source calibration interventions | Final-output calibration attribution |
| 2 | next8-16h | Implement/test B, full-video VAE and matched-budget controls | Decoder/state-space comparison |
| 3 | next8-16h | Public compatible cohort and optional external editor | Second-source task evidence |
| 4 | remaining12-24h | Failed-case recovery, blind-rating aggregation, manuscript | Tables/figures with audited claims |

Plan for roughly2-3 days INCLUDING implementation, testing, data staging and
recovery. These are allocation envelopes, not measured job-duration guarantees.
Existing17-frame RAE generation is about189s/video:32x4x2/24x189 is approximately
34 idealized minutes of sampling. Startup, unbalanced shards, codec backprop,
offline dependencies and publication dominate uncertainty; do not describe
these small comparisons as multi-day model training. Re-estimate after smoke.

If only48h remain: preserve repair, A, native-VAE operator control and blinded
ratings; reduce secondary seeds/public subset according to a predeclared schedule,
not observed outcome. Keep missing public/external evidence explicit. Do not
consume the last day on a new DINO-RAE training run or new geometry architecture.

## 4. Scope and execution requirements

Not required for the current narrowed story: retraining a larger AE/DiT, long
video, novel views,3D point clouds, WASD control, large-scale DINO-RAE training,
or isolating every reconstruction trick. To claim VGGT-specific geometry,
independent geometry and matched encoder controls WOULD become necessary.

New stages must retain original checkpoint/source hashes, fresh namespaces,
case-level resume, explicit fallback/failure counts, intermediate states/videos,
both-root media reconciliation, DI_throughput with units and no cache-hit inflation,
and the existing idle guard during CPU/staging periods. No outcome threshold may
block independent subsequent arms. Smoke checks correctness/memory, not a score.

The human task needs raters; external-model weights may need offline preparation.
No additional scene/depth/segmentation training dataset is required for A/B.
No new cluster execution, new training launcher, or new empirical result is
claimed by this planning update.
