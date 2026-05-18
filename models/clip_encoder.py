import torch
import torch.nn as nn


class CLIPTextEncoder(nn.Module):
    """CLIP ViT-L/14 text encoder. Frozen, runs on CPU by default, returns [B, L, 768]."""

    def __init__(self, model_name="/cache/yexiaoyu/vggae_ref/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"):
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = 768

    @torch.no_grad()
    def forward(self, texts):
        """texts: list of strings. Returns: [B, L, 768] float32."""
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=77, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.to(torch.float32)  # [B, L, 768]
