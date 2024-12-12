import numpy as np

## 1. base matmul
a = np.ones((3, 4)) * 2
b = np.ones((4, 5)) * 3

print("1. 2d matmul")
print(f"a{a.shape}:\n{a}\nb{b.shape}:\n{b}")
print(f"np.dot(a, b): {(a @ b).shape}")  # {3,5}

## 2. matmul with batch
c = np.ones((2, 3, 4)) * 2
d1 = np.ones(4) * 3
d2 = np.ones((4, 5)) * 3

# """
# - If `a` is an N-D array and `b` is a 1-D array, it is a sum product over
#   the last axis of `a` and `b`.
# - If `a` is an N-D array and `b` is an M-D array (where ``M>=2``), it is a
#   sum product over the last axis of `a` and the second-to-last axis of
#   `b`::
# """

print("2. 2d matmul w/ batch")

print(f"c{c.shape}:\n{c}\nd{d2.shape}:\n{d2}")
print(f"np.dot(c, d): {(c @ d2).shape}")  # {3,5}

# print(c @ d1)
# print(c @ d2)

einsum_result = np.einsum("abc,cd->abd", c, d2)
print(">>> einsum (btc,cd->btd)")
print(c @ d2 == einsum_result)
## 3. matmul with arbitrary shape
batch_size = 8
channel_size = 2
hidden_size = 32
other_dims = np.random.randint(1, 5, (np.random.randint(5)))


weight = np.ones((channel_size, hidden_size)) * 3
bias = np.zeros((hidden_size,))
arb_input = np.ones((batch_size, channel_size, *other_dims)) * 2

print(f"input: {arb_input.shape}")
print(f"weight: {weight.shape}")
print(f"bias: {bias.shape}")


def move_channel_dim_to_last(x: np.ndarray, channel_dim: int) -> np.ndarray:
    shape_tuple = x.shape
    if channel_dim == len(shape_tuple) - 1:
        return x
    x = np.expand_dims(x, -1)
    return np.squeeze(np.swapaxes(x, channel_dim, -1), axis=channel_dim)


linear_output = move_channel_dim_to_last(arb_input, 1)
linear_output = (linear_output @ weight) + bias
linear_output = np.swapaxes(np.expand_dims(linear_output, 1), 1, -1).squeeze(axis=-1)
print(f"output: {linear_output.shape}")

from torch import nn
import torch

model = nn.Linear(channel_size, hidden_size, bias=False)
model.weight = nn.Parameter(torch.FloatTensor(weight.T))
model.eval()

model_output = model(torch.FloatTensor(move_channel_dim_to_last(arb_input, 1)))
model_output = model_output.permute(0, -1, *range(1, len(arb_input.shape) - 1))
print(f"torch: {model_output.shape}")
