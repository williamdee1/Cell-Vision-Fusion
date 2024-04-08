# This work is made available under the Creative Commons Corporation CC BY 4.0 Legal Code.
# To view a copy of this license, visit
# https://github.com/williamdee1/Cell-Vision-Fusion/LICENSE.txt

import torch
import torch.nn as nn
import logging


def gpu_init_dp(model, criterion):
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        logging.info("Using %s DP GPU(s)..." % gpu_count)
        dev_ids = list(range(0, torch.cuda.device_count()))
        para_model = nn.DataParallel(model, device_ids=dev_ids)    # Create parallelized model
        device = 0
        para_model.to(device)                                      # Assigning model output to device:0
        criterion.to(device)

        return para_model, criterion, device

    else:
        model.cuda()
        criterion.cuda()
        device = "cuda:0"

        return model, criterion, device


