import torch
from torch import nn
from torch.nn import functional as F

from util.img2col import img2col_tensor


class CNN2DImplemented(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        pad_size: int = 0,
        stride: int = 1,
        debug: bool = False,
    ):
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._kernel_size = kernel_size
        self._pad_size = pad_size
        self._stride = stride
        if debug:
            self._weight = nn.Parameter(torch.ones((in_channel, kernel_size, kernel_size, out_channel)))
        else:
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
        output = torch.matmul(col, self._weight.reshape(-1, self._out_channel))
        return output.permute(0, -1, 1, 2)  # reshape to {N, C, H, W}


if __name__ == "__main__":
    # for debug output is correct
    c, h, w = 1, 7, 7
    img = [[[int(f"{k}{i}{j}") for i in range(1, w + 1)] for j in range(1, h + 1)] for k in range(1, c + 1)]
    # img = torch.rand(1, 3, 7, 7)
    img = torch.Tensor(img).unsqueeze(0)
    layer = CNN2DImplemented(c, c, 3, 1, debug=True)
    layer(img)
