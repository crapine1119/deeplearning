import torch
from torch import nn


class RNNImplemented(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._hidden = nn.Parameter(torch.zeros(out_channel))
        self._linear_ih = nn.Linear(in_channel, out_channel)
        self._linear_hh = nn.Linear(out_channel, out_channel)
        self._act = nn.Tanh()

    def forward(self, x: torch.Tensor):
        assert len(x.size()) == 3, AssertionError(f"Size of tensor should be 3")
        b, t, c = x.size()
        output = [self._hidden]
        for i in range(t):
            hidden_next = self._act(self._linear_ih(x[:, i]) + self._linear_hh(output[-1]))
            output.append(hidden_next)

        x_out = torch.stack(output[1:])
        return x_out.permute(1, 0, 2), output[-1]


if __name__ == "__main__":
    time_series = torch.randn((8, 100, 16))  # B, T, D
    model = RNNImplemented(16, 32)
    output, last_hidden = model(time_series)
    print(1)
