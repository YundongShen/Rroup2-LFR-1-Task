"""Pre-download OpenCLIP weights into the HF cache so training runs offline."""
import open_clip

for arch, tag in [("ViT-B-32", "laion2b_s34b_b79k"), ("ViT-B-32", "openai")]:
    open_clip.create_model_and_transforms(arch, pretrained=tag)
    print("ok", arch, tag)
