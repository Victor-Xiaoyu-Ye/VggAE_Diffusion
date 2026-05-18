"""Quick test: load Wan + decoder ckpt, run sampling + decode, save frames."""
import torch, os, numpy as np
from PIL import Image as PImage
from models.wan_adapter import WanVGGTAdapter
from utils.decoder_loader import load_decoder
from data.token_utils import build_decoder_tokens_from_generated

device = torch.device("cuda:0" if torch.npu.is_available() else "cpu")

# Load checkpoint (only model+ema, skip optimizer)
ckpt_path = "ckpts/diffusion_wan/exp-3-text/checkpoint_epoch0019.pt"
print(f"Loading {ckpt_path}...")
raw = torch.load(ckpt_path, map_location="cpu")
state = raw["model"]; ema_state = raw["ema"]; del raw
torch.npu.empty_cache()

# Build model
has_lora = any("lora_A" in k for k in state.keys())
model = WanVGGTAdapter(
    "Wan2.1/checkpoints/Wan2.1-T2V-1.3B",
    lora_rank=64 if has_lora else 0,
).to(device)
model.load_state_dict(state, strict=has_lora)
model.load_state_dict(ema_state, strict=False)
del state, ema_state; torch.npu.empty_cache()
model.eval()
print(f"  Model: {sum(p.numel() for p in model.parameters())/1e9:.2f}B, GPU: {(torch.cuda.memory_allocated() if torch.cuda.is_available() else 0)/1e9:.1f}GB")

# Load decoder
decoder = load_decoder("ckpts/decoder_dpt/exp-5-dpt/decoder_final.pt", device)
decoder.eval()

# ODE sampling (full 1369 patches)
N, S = 1369, 8
z = torch.randn(1, S, N, 2048, device=device).float()
num_steps = 20; dt = 1.0 / num_steps
print(f"Sampling ({S*N} tokens, {num_steps} steps)...")
with torch.no_grad():
    for i in range(num_steps):
        t_val = torch.tensor([i / num_steps], device=device)
        with torch.amp.autocast(device_type='npu', dtype=torch.float16):
            v = model(z, t_val)
        z = (z + v * dt).float()
print(f"  Done. z mean={z.mean():.4f} std={z.std():.4f}")

# Decode to RGB
print("Decoding...")
tokens = build_decoder_tokens_from_generated(z.float(), [11], seq_len=S)
dummy = torch.zeros(1, S, 3, 518, 518, device=device)
with torch.no_grad(), torch.amp.autocast(device_type='npu', dtype=torch.float16):
    result = decoder(tokens, images=dummy, patch_start_idx=0, frames_chunk_size=S)
if getattr(decoder, 'output_depth', False):
    preds, _, _, _ = result
else:
    preds, _ = result
recon = preds.permute(0, 1, 4, 2, 3).contiguous().clamp(0, 1)

# Save
out = "test_eval_out"; os.makedirs(out, exist_ok=True)
for s in range(S):
    f = (recon[0, s].float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    PImage.fromarray(f).save(f"{out}/frame_{s:02d}.png")
print(f"Saved {S} frames to {out}/")
