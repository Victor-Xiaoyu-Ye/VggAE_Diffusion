#!/bin/bash
# Migrate all Python files from CUDA to Ascend NPU (910B)
# Run this script ONCE on the ascend-910b branch
set -e

PROJECT=/home/yexiaoyu/work/VggAE-Diffusion
cd $PROJECT

echo "=== Migrating to Ascend 910B ==="

# ---- Python files ----
# 1. autocast device_type
find . -name "*.py" -not -path "./ckpts/*" -not -path "./samples/*" -not -path "./analysis/*" -not -path "./reconstruct_out/*" -not -path "./.git/*" -exec sed -i \
  -e 's/device_type="cuda"/device_type="npu"/g' \
  -e "s/device_type='cuda'/device_type='npu'/g" \
  {} +

# 2. torch.cuda → torch.npu (functions)
find . -name "*.py" -not -path "./ckpts/*" -not -path "./samples/*" -not -path "./analysis/*" -not -path "./reconstruct_out/*" -not -path "./.git/*" -exec sed -i \
  -e 's/torch\.cuda\.is_available/torch.npu.is_available/g' \
  -e 's/torch\.cuda\.empty_cache/torch.npu.empty_cache/g' \
  -e 's/torch\.cuda\.manual_seed_all/torch.npu.manual_seed_all/g' \
  -e 's/torch\.cuda\.amp/torch.npu.amp/g' \
  -e 's/torch\.cuda\.current_device/torch.npu.current_device/g' \
  -e 's/torch\.cuda\.device_count/torch.npu.device_count/g' \
  {} +

# 3. torch.device("cuda") → torch.device("npu") — keep the f-string version for later
find . -name "*.py" -not -path "./ckpts/*" -not -path "./samples/*" -not -path "./analysis/*" -not -path "./reconstruct_out/*" -not -path "./.git/*" -exec sed -i \
  -e 's/torch\.device("cuda"/torch.device("npu"/g' \
  -e "s/torch\.device('cuda'/torch.device('npu'/g" \
  -e 's/torch\.device(f"cuda:/torch.device(f"npu:/g' \
  -e "s/torch\.device(f'cuda:/torch.device(f'npu:/g" \
  {} +

# 4. .to("cuda") → .to("npu") (but NOT .to(device) patterns)
find . -name "*.py" -not -path "./ckpts/*" -not -path "./samples/*" -not -path "./analysis/*" -not -path "./reconstruct_out/*" -not -path "./.git/*" -exec sed -i \
  -e 's/\.to("cuda")/.to("npu")/g' \
  -e "s/\.to('cuda')/.to('npu')/g" \
  {} +

# 5. pin_memory=True → False (NPU doesn't support the same way)
find . -name "*.py" -not -path "./ckpts/*" -not -path "./samples/*" -not -path "./analysis/*" -not -path "./reconstruct_out/*" -not -path "./.git/*" -exec sed -i \
  -e 's/pin_memory=True/pin_memory=False/g' \
  {} +

# 6. GradScaler → disabled for NPU (NPU handles mixed precision natively)
find . -name "*.py" -not -path "./ckpts/*" -not -path "./samples/*" -not -path "./analysis/*" -not -path "./reconstruct_out/*" -not -path "./.git/*" -exec sed -i \
  -e 's/GradScaler(enabled=(not use_bf16))/GradScaler(enabled=False)/g' \
  {} +

# 7. bf16 → fp16 on NPU (Ascend 910B doesn't fully support bf16)
find . -name "*.py" -not -path "./ckpts/*" -not -path "./samples/*" -not -path "./analysis/*" -not -path "./reconstruct_out/*" -not -path "./.git/*" -exec sed -i \
  -e 's/dtype=torch\.bfloat16/dtype=torch.float16/g' \
  -e "s/dtype=torch\.float32/dtype=torch.float32/g" \
  {} +

# 8. bf16 autocast → fp16
find . -name "*.py" -not -path "./ckpts/*" -not -path "./samples/*" -not -path "./analysis/*" -not -path "./reconstruct_out/*" -not -path "./.git/*" -exec sed -i \
  -e "s/autocast(device_type='npu', dtype=torch.bfloat16)/autocast(device_type='npu', dtype=torch.float16)/g" \
  -e 's/autocast(device_type="npu", dtype=torch.bfloat16)/autocast(device_type="npu", dtype=torch.float16)/g' \
  {} +

# ---- Shell scripts ----
find . -name "*.sh" -not -path "./ckpts/*" -not -path "./samples/*" -exec sed -i \
  -e 's/ASCEND_RT_VISIBLE_DEVICES/ASCEND_RT_VISIBLE_DEVICES/g' \
  {} +

echo "Done. Review changes with: git diff"
echo "Then: git add . && git commit -m 'migrate to Ascend 910B (cuda→npu)'"
