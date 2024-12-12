import torch
from torch import nn
from torch.nn import functional as F


class CNN2DImplemented(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int = 3, pad_size: int = 0, stride: int = 1):
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._kernel_size = kernel_size
        self._pad_size = pad_size
        self._stride = stride
        self._weight = nn.Parameter(torch.randn((kernel_size, kernel_size)))
        self._bias = nn.Parameter(torch.randn((kernel_size, kernel_size)))

    def _reshape_img(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self._pad_size, self._pad_size, self._pad_size, self._pad_size, 0, 0, 0, 0)
        return F.pad(x, pad)[..., :: self._stride, :: self._stride]

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 4, AssertionError(
            f"length of tensor should be (B,C,H,W), but current length is {len(x)}"
        )
        x = self._reshape_img(x)
        return


if __name__ is "__main__":
    img = torch.rand(4, 3, 7, 7)
    layer = CNN2DImplemented(16, 3, 1, 1)
    layer(img)
