"""Small tensor ops shared by the A-only method and the pilot analysis."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import RSICDImages


def d4(x, g):
    """Element g in 0..7 of the dihedral group D4 on image batches [B, C, H, W]: rot90^(g%4), then hflip if g>=4.
    Overhead imagery has no canonical orientation, so semantics are (approximately) D4-invariant."""
    x = torch.rot90(x, g % 4, dims=(2, 3))
    return x.flip(3) if g >= 4 else x


@torch.no_grad()
def sinkhorn(scores, eps=0.05, n_iter=3):
    """Balanced soft assignment (SwAV-style): scores [N, C] -> Q [N, C], rows sum to 1, columns ~ N/C."""
    Q = torch.exp((scores - scores.max()) / eps).T.double()
    Q /= Q.sum()
    C, N = Q.shape
    for _ in range(n_iter):
        Q /= Q.sum(1, keepdim=True) * C
        Q /= Q.sum(0, keepdim=True) * N
    return (Q * N).T.float()


@torch.no_grad()
def encode_images_d4(model, df, transform, dev, bs=256, num_workers=8, group=range(8)):
    """[N, len(group), d] normalized image features of every D4 view."""
    out = []
    for batch in DataLoader(RSICDImages(df, transform), batch_size=bs, num_workers=num_workers):
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(dev)
        out.append(torch.stack([F.normalize(model.encode_image(d4(x, g)).float(), dim=-1) for g in group], 1).cpu())
    return torch.cat(out)
