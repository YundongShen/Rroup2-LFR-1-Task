class ZeroShot:
    """Baseline: pretrained CLIP without any adaptation."""

    trains = False

    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, model, tokenizer, train_tf, eval_tf, b_train, b_val, a_unlabeled, device, out_dir):
        return model
