# MOAB Method source: https://github.com/omniaalwazzan/MOAB/blob/main/moab_fusion_model.py

import torch
import torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



class ConvStack(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Conv_ = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),   ### fix it by tunning [1,3,7]
            #nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),   ### for k=3 pandding is 1
            #nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.Dropout(p=0.02)
            )

    def forward(self, x):
        return self.Conv_(x)


def append_0_s(x1, x3):
    """ Outer subtraction """
    b = torch.tensor([[0]]).to(device=DEVICE, dtype=torch.float32)
    x1 = torch.cat((b.expand((x1.shape[0], 1)), x1), dim=1)
    # print('this is x1 and this is the shape of x1',x1.shape)

    x3 = torch.cat((b.expand((x3.shape[0], 1)), x3), dim=1)
    # print('this is x1 and this is the shape of x1',x3.shape)

    x_p = x3.view(x3.shape[0], x3.shape[1], 1) - x1.view(x1.shape[0], 1, x1.shape[1])
    x_p = torch.sigmoid(x_p)
    # print('the shape of xp after outer add bfr flatten',x_p.shape)
    # x_p = x_p.flatten(start_dim=1)
    return x_p


def append_0(x1, x3):
    """ Outer addition """
    b = torch.tensor([[0]]).to(device=DEVICE, dtype=torch.float32)
    x1 = torch.cat((b.expand((x1.shape[0], 1)), x1), dim=1)
    # print('this is x1 in add and this is the shape of x1',x1.shape)

    x3 = torch.cat((b.expand((x3.shape[0], 1)), x3), dim=1)
    # print('this is x1 and this is the shape of x1',x3.shape)

    x_p = x3.view(x3.shape[0], x3.shape[1], 1) + x1.view(x1.shape[0], 1, x1.shape[1])
    x_p = torch.sigmoid(x_p)
    # print('the shape of xp after outer add bfr flatten',x_p.shape)
    # x_p = x_p.flatten(start_dim=1)
    return x_p


def append_1(x1, x3):
    """ Outer product """
    b = torch.tensor([[1]]).to(device=DEVICE, dtype=torch.float32)
    x1 = torch.cat((b.expand((x1.shape[0], 1)), x1), dim=1)
    # print('this is x1 of OP and this is the shape of x1',x1.shape)

    x3 = torch.cat((b.expand((x3.shape[0], 1)), x3), dim=1)
    # print('this is x1 and this is the shape of x1',x3.shape)

    x_p = x3.view(x3.shape[0], x3.shape[1], 1) * x1.view(x1.shape[0], 1, x1.shape[1])
    x_p = torch.sigmoid(x_p)
    # print('the shape of xp after outer pro bfr flatten',x_p.shape)
    # x_p = x_p.flatten(start_dim=1)
    return x_p


def append_1_d(x1, x3):
    """ Outer division """
    b = torch.tensor([[1]]).to(device=DEVICE, dtype=torch.float32)
    x1 = torch.cat((b.expand((x1.shape[0], 1)), x1), dim=1)
    # print('this is x1 of div and this is the shape of x1',x1.shape)

    x3 = torch.cat((b.expand((x3.shape[0], 1)), x3), dim=1)

    x1_ = torch.full_like(x1, fill_value=float(1.2e-20))
    x1 = torch.add(x1, x1_)

    # print('this is x1 and this is the shape of x1',x3.shape)

    x_p = x3.view(x3.shape[0], x3.shape[1], 1) / x1.view(x1.shape[0], 1, x1.shape[1])
    x_p = torch.sigmoid(x_p)
    # print('the shape of xp after outer pro bfr flatten',x_p.shape)
    # x_p = x_p.flatten(start_dim=1)
    return x_p
