from torch.optim.lr_scheduler import SequentialLR
import torch as th


def get_cosine_warmup_scheduler(optimizer, warmup_steps, cosine_steps):
    warmup_lr = th.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: step / warmup_steps
    )
    cosine_lr = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps)
    return SequentialLR(
        optimizer, schedulers=[warmup_lr, cosine_lr], milestones=[warmup_steps]
    )
