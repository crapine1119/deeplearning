import torch
from torch import nn
from torch.nn import functional as F

from util.img2col import img2col_tensor


class CNN2DImplemented(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int = 3, pad_size: int = 0, stride: int = 1):
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._kernel_size = kernel_size
        self._pad_size = pad_size
        self._stride = stride
        self._weight = nn.Parameter(torch.randn((in_channel, kernel_size, kernel_size, out_channel)))
        self._bias = nn.Parameter(torch.randn((out_channel, kernel_size, kernel_size)))

    def _pad_img(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self._pad_size, self._pad_size, self._pad_size, self._pad_size, 0, 0, 0, 0)
        return F.pad(x, pad)

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 4, AssertionError(
            f"length of tensor should be (B,C,H,W), but current length is {len(x)}"
        )
        x = self._pad_img(x)
        col = img2col_tensor(x, kernel_size=self._kernel_size, stride=self._stride)
        col = torch.matmul(col, self._weight.reshape(-1, self._out_channel))
        return torch.dot(col, self._weight.reshape(-1, self._out_channel))


if __name__ == "__main__":
    img = torch.rand(4, 3, 7, 7)
    layer = CNN2DImplemented(3, 16, 3, 1)
    layer(img)
