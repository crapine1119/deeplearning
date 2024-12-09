import numpy as np

## 1. base matmul
a = np.ones((3, 4)) * 2
b = np.ones((4, 5)) * 3

print(">>> 2d matmul")
print(np.dot(a, b) == a @ b)  # {3,5}

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

print(">>> 2d w/ batch")
print(c @ d1)
print(c @ d2)

einsum_result = np.einsum("abc,cd->abd", c, d2)
print(">>> einsum (btc,cd->btd)")
print(c @ d2 == einsum_result)
## 3. matmul with arbitrary shape
e = np.random.rand()
