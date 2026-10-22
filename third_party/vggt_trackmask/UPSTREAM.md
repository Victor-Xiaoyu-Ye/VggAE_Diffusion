# Official VGGT diagnostic teacher source

Source: https://github.com/facebookresearch/vggt

Commit: `a288dd0f14786c93483e45524328726ab7b1b4ce`

Retrieved 2026-10-22. The `vggt/models`, `vggt/heads`, `vggt/layers`,
`vggt/utils/pose_enc.py`, and `vggt/utils/rotation.py` source files are copied
unchanged. The accompanying `LICENSE.txt` governs upstream files. No weights,
demo dependencies, data, network loader, or production encoder changes included.

Runtime dependencies: torch, numpy, einops, huggingface_hub; safetensors for
`.safetensors` weights. Do not replace cluster torch/torch_npu with upstream's
historical pinned requirements.

Explicit full VGGT camera, depth, and track weights are required. The adapter
checks every active state key and shape; StreamVGGT/Omega/aggregator-only weights
must not be silently accepted. Official model API is called through its separate
heads to enforce FP32 head arithmetic on CPU/CUDA/NPU. NPU execution still needs
cluster validation. Query chunking changes the tracker's query interaction
context and is recorded as a separate diagnostic configuration.

Official weight metadata checked without downloading weights:

- Repository: `facebook/VGGT-1B`
- Revision: `860abec7937da0a4c03c41d3c269c366e82abdf9`
- `model.safetensors`: 5026367224 bytes; SHA256
  `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`
- `model.pt`: 5026874952 bytes; SHA256
  `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`

Source metadata: https://huggingface.co/api/models/facebook/VGGT-1B?blobs=true

This teacher sees all supplied frames and only provides offline diagnostic
pseudo-labels. Full-window masks must never be passed to a history-only
generation model as if they were available at inference.
