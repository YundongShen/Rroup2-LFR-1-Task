import open_clip
import torch


def get_device(name="auto"):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def build_model(model_cfg, device):
    """Returns (model, train_transform, eval_transform, tokenizer)."""
    model, train_tf, eval_tf = open_clip.create_model_and_transforms(
        model_cfg.arch, pretrained=model_cfg.pretrained, precision=model_cfg.precision, device=device)
    return model, train_tf, eval_tf, open_clip.get_tokenizer(model_cfg.arch)
