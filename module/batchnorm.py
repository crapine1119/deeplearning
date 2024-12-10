from enum import Enum, auto

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import BatchNorm2d


class TrainingMode(Enum):
    train = auto()
    test = auto()


class BatchNorm2DImplemented(nn.Module):
    def __init__(self, channel_size: int, mode: TrainingMode = TrainingMode.train):
        super().__init__()
        self._gamma = nn.Parameter(torch.ones((channel_size,)))
        self._beta = nn.Parameter(torch.zeros((channel_size,)))
        self._momentum = 0.1
        self._mean = 0
        self._var = 0
        self._eps = 1e-12
        self._mode = TrainingMode.train

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._mode == TrainingMode.train:
            batch_size = x.size(0)
            self._mean = self._momentum * self._mean + (1 - self._momentum) * x.mean(dim=0)
            self._var = self._momentum * self._var + (1 - self._momentum) * x.var(dim=0) * batch_size / (batch_size - 1)
            return (x - x.mean(dim=0)) / ((self._var + self._eps) ** 0.5)
        else:
            return (x - self._mean) / (self._var + self._eps)

    def forward(self, x: torch.Tensor):
        x_normalized = self._normalize(x)
        if self._mode == TrainingMode.train:
            x_distributed = torch.einsum("bchw,c->bchw", x_normalized, self._gamma)
            x_distributed = x_distributed.transpose(1, -1) + self._beta
            x_distributed = x_distributed.transpose(1, -1)
        else:
            with torch.no_grad():
                x_distributed = x_normalized.transpose(0, -1) * self._gamma + self._beta
                x_distributed = x_distributed.transpose(0, -1)
        return x_distributed


if __name__ == "__main__":
    ##
    b, c, h, w = 64, 8, 7, 7
    input_tensor = torch.randn((b, c, h, w)) + 10
    # implemented batch norm
    layer = BatchNorm2DImplemented(c, TrainingMode.train)
    implemented = layer(input_tensor)

    # origin batch norm
    origin = BatchNorm2d(c)(input_tensor)
    hist_bin = 20
    feature_index = 7
    fig = plt.figure()
    ax1, ax2 = fig.subplots(2, 1)
    ax1.hist(input_tensor[:, feature_index, 0, 0].tolist(), bins=hist_bin, label="input")
    ax1.grid()
    ax1.legend(loc=1)
    ax2.hist(implemented[:, feature_index, 0, 0].tolist(), bins=hist_bin, label="implemented bn")
    ax2.hist(origin[:, feature_index, 0, 0].tolist(), bins=hist_bin, label="origin bn")
    ax2.grid()
    ax2.legend(loc=1)
    plt.show()
