"""Exponential moving average teacher of the adapter weights."""
import torch

from src.experiment.residual import AdaptedCLIP


class EMATeacher:
    """A second AdaptedCLIP around the same frozen CLIP whose adapters track the student.

        teacher <- momentum * teacher + (1 - momentum) * student

    after every optimizer step. The teacher never receives gradients; its outputs
    are the consistency targets and, optionally, the pseudo-labels.
    """

    def __init__(self, student, momentum=0.99):
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        self.momentum = float(momentum)
        self.model = AdaptedCLIP(
            student.clip, hidden_dim=student.image_adapter.fc1.out_features if student.image_adapter is not None
            else student.text_adapter.fc1.out_features,
            alpha=student.alpha, beta=student.beta, adapters=student.adapters, learn_logit_scale=False)
        self.model.to(next(student.parameters()).device)
        self.model.load_state_dict(student.state_dict())
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def update(self, student):
        m = self.momentum
        teacher = dict(self.model.named_parameters())
        for name, s in student.named_parameters():
            if name in teacher:
                teacher[name].mul_(m).add_(s.detach(), alpha=1.0 - m)
        # The temperature is shared, not averaged: the teacher scores with the student's.
        self.model.logit_scale.copy_(student.logit_scale.detach())

    def state_dict(self):
        return self.model.state_dict()
