# Stage70: calibration attribution and common-generator operator evidence

## Purpose

Close two concrete gaps in the paper: distinguish initial decoder calibration
from online solving, and separate the readout operator from separately trained
state bridges. This is inference plus latent-state fitting, not new AE, LoRA,
or DiT training. No empirical improvement is assumed.

## Launch on all three nodes

Use the existing repository/reference bootstrap, then run the same commands on
all 3 nodes (8 Ascend NPUs each):

```bash
ATTR_STAGE=smoke bash scripts/scale/70_audit_edit_attribution.sh &&
ATTR_STAGE=all bash scripts/scale/70_audit_edit_attribution.sh
```

Default output: `output/scale/editing_attribution_v1`; smoke has `_smoke` suffix.
The unchanged parent is `editing_paper_evidence_v1`, even if its downloaded
export is named `editing_paper_evidence_v2`. State weights come from
`vace_edit_state_v1`; reference-SFT weights and shared references come from the
parent. The existing `vace_edit_pilot_v1` bank supplies evaluation metadata.
Full reference assets and parent caches must be readable through existing roots.

```bash
# Continue after interruption; completed generation is reused.
ATTR_STAGE=resume bash scripts/scale/70_audit_edit_attribution.sh
# Regenerate only evaluation/reporting, without sampling:
ATTR_STAGE=evaluate bash scripts/scale/70_audit_edit_attribution.sh
```

Use a fresh `ATTR_NAMESPACE` for changed code/config/weights. Resume excludes
failed feedback from successful means and retries those cases. A phase failure
does not stop independent arms; final exit status remains nonzero when incomplete.
The `&&` intentionally prevents launching full work after a failed cluster smoke.

## Fixed protocol

- Existing Stage69 32-source ablation subset, seed 42 with the existing per-case
  deterministic offset, 17 frames, 8 fps, 30 sampling steps. All 14 arms: 448
  outputs. Smoke: 3 sources, one per node, 6 sampling steps, 42 outputs.
- Same source, shared model-generated reference, prompt and native context.
  Paired edited targets are accessed only during evaluation.
- Frozen learned RAE and generation weights. Initial states are re-encoded
  from the shared proposal so the uncalibrated branch is genuinely uncalibrated.
- Initial fitting and online solving are separate interventions; both are
  source/reference-only. The trained RAE remains part of the method history.

| Group | Arms | Question |
|---|---|---|
| P0 main | Initial calibration on/off × online solve on/off | Which stage causes final-output changes? |
| P0 bridge control | Same four settings, calibrated bridge fixed throughout | Is the effect explained by changing the learned bridge input? |
| P0 magnitude | Uncalibrated reuse scaled to calibrated correction RMS at the same current prediction | Is correction magnitude sufficient to explain the difference? |
| P1 | SFT alone, RGB, spatially smoothed RGB, full-video native VAE, RAE | Does a decodable readout help under the same SFT generator without a state bridge? |

P1 uses cached initial effects and the same visibility gate, hard anchor,
writeback schedule, alpha and RMS cap. It does not perform online solving.
VAE fits the full native 17-frame temporal latent, with its native normalization;
it is not a framewise VAE proxy. VAE fitting runs before loading the 14B model.
RAE retains its production nine-observation lattice and source-flow expansion.
Therefore this compares practical operators, not identical tokenizer capacity
or temporal lattices. Plain RGB is an essential control and is retained.

The spatial Gaussian sigma is predeclared as 2 pixels, not bandwidth-matched.
Schedules are matched, wall time is not; fitting/setup/sampling costs are saved.
Magnitude matching logs achieved RMS after clipping, which can differ from the
requested RMS. It incurs an extra calibrated decoder readout and is a diagnostic.

## Outputs and interpretation

- `calibration/<id>.pt/json`: raw/fitted states, fitting diagnostics, parent
  state deviation, initial delta amplitudes and visible RAE effects.
- `vae/<id>.pt/json`: full-video effects, fitting objective and gradient norms.
- `render/<arm>/seed42/<id>/`: generated video/NPZ, metrics/provenance receipt,
  per-event correction videos and state snapshots.
- `summary.json`: separate P0/P1 tables, source-group paired bootstrap intervals,
  coverage/failures and timing/memory records. The internal `rae_full` table key
  aliases the explicitly recorded `reference_arm`; it is not the old Stage69 arm.
- `media_sync.json`: final and correction video receipt checks on both roots.
- Per-rank progress includes `DI_throughput` with units; cache hits do not count
  as generated frames. Existing idle guard, logs and coordinator remain active.

No new benchmark or human-rating claim comes from this suite. If initial
calibration has no measurable benefit, narrow the calibration claim; if common-
SFT RGB matches RAE, do not attribute the old result solely to its readout.
Public-cohort generalization remains the next pending experiment, after this
mechanistic attribution. Report the complete fixed cohort, not selected cases
as an aggregate. Strong qualitative examples may accompany these full tables.

## Validation boundary

Local tests cover source-only operators, full-video latent gradients on a small
codec and reduced-width native Wan causal VAE with activation recomputation,
actual tiny VACE sampling with live writeback, prefix preservation,
rollback, intervention state isolation, and real Bash stage/failure/cleanup
execution using bounded process stubs. They do not validate production NPU
memory, native 910B VAE autograd, HCCL or OBS. The three-node smoke does that.
