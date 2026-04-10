import torch
from torch import nn
from tools.models import ModelLoader  # adjust to your repo layout

class VolumeEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        full_model = ModelLoader.load_full_prima_model(config)
        visual_model = full_model.clipvisualmodel   # HierViT
        self.inner_vit = visual_model.innerViT
        self.device = next(self.inner_vit.parameters()).device

    @torch.no_grad()
    def forward(self, series_tokens: torch.Tensor, lens: torch.Tensor):
        series_tokens = series_tokens.to(self.device)
        lens = lens.to(self.device)
        xdict = {"visual": series_tokens, "lens": lens}
        emb, _ = self.inner_vit(xdict, retboth=True)
        return emb

if __name__ == "__main__":
    # usage
    cfg = {...}  # same config you use for FullMRIModel
    enc = VolumeEncoder(cfg)

    tokens = ...  # [1, T, D]
    lens = torch.tensor([T])
    vol_emb = enc(tokens, lens)  # [1, clsnum*dim]