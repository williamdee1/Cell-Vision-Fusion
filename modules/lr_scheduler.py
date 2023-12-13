# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# Written by Ze Liu
# Adapted from Source: https://github.com/microsoft/Swin-Transformer/blob/main/lr_scheduler.py
# --------------------------------------------------------

import bisect

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_scheduler(args, optimizer):
    """
    Swin Config - https://github.com/microsoft/Swin-Transformer/blob/f92123a0035930d89cf53fcb8257199481c4428d/config.py#L170
    """
    lr_scheduler = None
    if args.sched_type == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=args.t_initial,
            cycle_mul=args.cycle_mul,
            lr_min=args.min_lr,
            cycle_decay=args.lrd_fac,
            warmup_t=args.warmup_epochs,
            warmup_lr_init=args.warmup_lr,
            warmup_prefix=True,
            cycle_limit=args.cycle_limit,
            t_in_epochs=True,
        )
    elif args.sched_type == 'linear':
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            factor=args.lrd_fac,
            patience=args.lr_pat,
            verbose=False,
            mode='min',
            min_lr=args.min_lr)

    return lr_scheduler

