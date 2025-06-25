import torch
import torch.nn as nn
import logging
class ClippedReLU(nn.Module):
    def __init__(self, lower_bound=0.0, upper_bound=1.0):
        super(ClippedReLU, self).__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, x):
        y = torch.zeros_like(x)
        y[x < self.lower_bound] = 0.001*(x[x < self.lower_bound] - self.lower_bound)
        y[x > self.upper_bound] = 0.001*(x[x > self.upper_bound] - self.upper_bound) + self.upper_bound
        y[(x >= self.lower_bound) & (x <= self.upper_bound)] = x[(x >= self.lower_bound) & (x <= self.upper_bound)]
        return y

    def forward_1(self, x):
        y = (1+torch.tanh(x))/2
        return y




class PolyAct(nn.Module):
    def __init__(self, degree=3, lower_bound=0.0, upper_bound=1.0):
        super(PolyAct, self).__init__()
        self.degree = degree
        self.clip_relu = ClippedReLU(lower_bound, upper_bound)

    def forward(self, x):
        B, Q, C = x.shape
        x = x.view(B, Q, 2, self.degree+1).contiguous()
        y = torch.zeros_like(x)
        d = self.clip_relu(x[:, :, :, -1])
        y[:, :, :, 0] = x[:, :, :, 0]
        y[:, :, :, 1:-1] = x[:, :, :, 1:-1]
        y[:, :, :, -1] = d
        y = y.view(B,  Q, C).contiguous()
        if torch.isnan(y).any():
            logging.error(f"x has nan at x: {x}")
            logging.error(f"y has nan at y: {y}")
        return y

class InvPolyAct(nn.Module):
    def __init__(self, degree=3, lower_bound=0.0, upper_bound=1.0):
        super(InvPolyAct, self).__init__()
        self.degree = degree
        self.clip_relu = ClippedReLU(lower_bound, upper_bound)

    def forward(self, x):
        B, Q, C = x.shape
        x = x.view(B*Q, 2, self.degree+1).contiguous()
        y = torch.zeros_like(x)
        d = self.clip_relu.unclip(x[:, :, -1])
        deg_sum = self.clip_relu.unclip(x.sum(dim=-1))
        if(torch.sum(deg_sum.isnan())>0):
            logging.error(f"deg_sum has at nan at x: {x.sum(dim=-1)}")
        y[:, :, 0] = deg_sum
        y[:, :, 1:-1] = x[:, :, 1:-1]
        y[:, :, -1] = d
        y = y.view(B,  Q, C).contiguous()
        return y