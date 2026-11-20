"""One function per task. A run calls exactly one of them."""
import torch
import torch.nn.functional as F

from src.data import class_prompt_names
from src.experiment.encode import caption_texts, encode_images, encode_texts, recall_at, retrieval_ranks


@torch.no_grad()
def classification(model, tokenizer, transform, df, device, batch_size, num_workers, classes, prompts, progress):
    """
    Classification task. Predict the class of the image from the matched prompt.
    Parameters:
    - model: the model to use
    - tokenizer: the tokenizer to tokenize the text inputs
    - transform: the transform to use for the images
    - df: the dataframe containing the images and labels
    - device: the device to use for the model
    - batch_size: the batch size to use for the model
    - num_workers: the number of workers to use for the data loader
    - classes: the classes to use for the classification
    - prompts: the prompts to use for the classification
    Returns:
    - A dictionary containing the accuracy and the mean per-class accuracy.
    """
    img_feats, labels = encode_images(
        model, df, transform, device, batch_size, num_workers, desc=f"{progress} images")
    names = class_prompt_names(classes, prompts.class_names)
    templates = list(prompts.templates)
    weight = []
    for name in names:
        embedding = encode_texts(
            model, tokenizer, 
            [t.format(name) for t in templates], device, batch_size
        ).mean(0) # average over the templates for the class, no effect if there is only one template
        weight.append(F.normalize(embedding, dim=-1)) # L2 normalize the text features, size is [d]
    # both image and text features are L2 normalized, so the dot product is the cosine similarity
    pred = (img_feats @ torch.stack(weight).T).argmax(1) # pred is of size [n_images], the index of the class with the highest cosine similarity
    present = labels.unique() # get the unique classes in the labels, size is [n_classes]
    per_class = torch.stack([(pred[labels == k] == k).float().mean() for k in present]) # compute the per-class accuracy, size is [n_classes]
    return {
        "acc": (pred == labels).float().mean().item(), # overall accuracy
        "mean_per_class_acc": per_class.mean().item(), # mean per-class accuracy
    }


@torch.no_grad()
def image_to_text(model, tokenizer, transform, df, device, batch_size, num_workers, ks, progress):
    img_feats, _ = encode_images(
        model, df, transform, device, batch_size, num_workers, desc=f"{progress} images")
    txt_feats = encode_texts(
        model, tokenizer, caption_texts(df), device, batch_size, desc=f"{progress} captions")
    i2t, _ = retrieval_ranks(img_feats, txt_feats) # ranks the similarity between the image and text features
    return recall_at(i2t, ks) # recall at k, size is [n_ks]


@torch.no_grad()
def text_to_image(model, tokenizer, transform, df, device, batch_size, num_workers, ks, progress):
    img_feats, _ = encode_images(
        model, df, transform, device, batch_size, num_workers, desc=f"{progress} images")
    txt_feats = encode_texts(
        model, tokenizer, caption_texts(df), device, batch_size, desc=f"{progress} captions")
    _, t2i = retrieval_ranks(img_feats, txt_feats)
    return recall_at(t2i, ks) # recall at k, size is [n_ks]


@torch.no_grad()
def mean_r1(model, tokenizer, transform, df, device, batch_size, num_workers, ks, progress):
    """Both retrieval directions on the same candidate pool.

    For every k in ks: i2t_r{k}, t2i_r{k} and mean_r{k} = (i2t_r{k} + t2i_r{k}) / 2.
    mean_r1 comes first and stays the headline / selection metric; mean_recall averages the
    six (or 2 * len(ks)) recall values, a smoother number than R@1 on a small eval fold.
    """
    img_feats, _ = encode_images(
        model, df, transform, device, batch_size, num_workers, desc=f"{progress} images")
    txt_feats = encode_texts(
        model, tokenizer, caption_texts(df), device, batch_size, desc=f"{progress} captions")
    i2t, t2i = retrieval_ranks(img_feats, txt_feats)
    ks = sorted(set(int(k) for k in ks) | {1})
    r_i2t, r_t2i = recall_at(i2t, ks), recall_at(t2i, ks)
    out = {}
    for k in ks:
        out[f"mean_r{k}"] = 0.5 * (r_i2t[f"R@{k}"] + r_t2i[f"R@{k}"])
        out[f"i2t_r{k}"] = r_i2t[f"R@{k}"]
        out[f"t2i_r{k}"] = r_t2i[f"R@{k}"]
    if len(ks) > 1:
        out["mean_recall"] = sum(out[f"mean_r{k}"] for k in ks) / len(ks)
    return out


TASKS = {
    "classification": classification,
    "i2t": image_to_text,
    "t2i": text_to_image,
    "mean_r1": mean_r1,
}
