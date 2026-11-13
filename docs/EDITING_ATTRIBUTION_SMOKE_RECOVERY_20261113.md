# Stage70 smoke: feedback preview call repair

## Observed production evidence

Local export: `D:/中转站/VGGAE/scale/editing_paper_evidence_v1_smoke`.
Actual runtime namespace: `editing_attribution_v1_smoke`.
Manifest identity:
`8c6671a190ddcdcd2169a6b5df13e4ec4157828988ff68e0c0e49a6be57bd397`.
All 21 recorded code signatures match the pre-fix checkout.

| Phase | Completed | Observation |
|---|---:|---|
| RAE initial calibration | 3/3 | No fallback; fitting objective decreases on all three |
| Full-video native VAE calibration | 3/3 | Finite nonzero gradients and lower fitting objective on all three |
| SFT without feedback | 3/3 | Final video and evaluation complete |
| 13 feedback arms | 0/39 | Identical `write_video() missing 1 required positional argument: 'fps'` |
| Evaluation | 3/42 | Remaining cases have no final committed result |
| Summary | incomplete | Missing results are correctly rejected; `&&` does not start full run |

The early Scene C256 local checkpoint FileNotFoundError is followed by successful
OBS fallback and stage exit 0. It is not the cause of the generation failures.
The 39 failed generations all enter the feedback path, save a state snapshot,
and then raise while writing its preview. There is no completed RAE/RGB/VAE final
video comparison to draw a quality conclusion from this smoke.

## Root cause and patch

`utils.editing_train.write_video` expects `(path, source, target, generated, fps)`.
Stage70 passed `(path, source, correction, fps)`: the fps integer occupied the
generated argument and the required fps argument was absent. Restore the shared
reference panel, keep source-only preview content, and pass `fps` by keyword.
All panels use the same valid crop and uint8 conversion, so the fix also preserves
the established aspect-ratio and pixel-range contract.

The previous local tests exercised operators and sampling without Stage70's own
preview observer. They could not detect this call-site bug. New integration tests
execute the real driver, solver, tiny VACE, callbacks, video writer, artifact IO,
evaluation and summary for every arm. Tests decode the actual MP4, inspect panel
dimensions and pixel values, compare hashes across two roots, repair missing
replicas, and prove completed results do not trigger another model load.

Only the reviewed driver before/after hashes are allowed to share the original
manifest. Every other dependency, config, cohort and weight identity must still
match. Prepared caches and the three completed SFT results keep their identities;
new results separately record the executing code signature. Phase summaries now
print root causes, while final reports retain both upstream failures and their
downstream missing-media consequences.

## Restart

Update the repository used by the cluster bootstrap. On all three nodes:

```bash
ATTR_STAGE=smoke bash scripts/scale/70_audit_edit_attribution.sh &&
ATTR_STAGE=all bash scripts/scale/70_audit_edit_attribution.sh
```

Keep the default `editing_attribution_v1` namespace. The first command recovers
`editing_attribution_v1_smoke`, reuses completed preparation/SFT, and fills the
39 missing outputs. The full 32-source run begins only after smoke is complete.
This patch changes media/reporting/recovery only, not the solver, backbone,
sampler, conditions, objectives or experiment matrix. Production completion and
comparative quality remain pending the rerun.

Validation: 46 local tests pass across Stage70 operators, the new driver/media
integration, Bash launcher, feedback, state-data and Stage69 regression suites.
Python compilation, Bash syntax and whitespace checks pass. The actual downloaded
production manifest passes the exact-hash migration; recomputing all three SFT
generation identities produces their original values. These local checks do not
claim that the repaired production cluster job has already completed.
