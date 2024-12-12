import numpy as np


def img2col(x: np.ndarray, kernel_size: int = 3, stride: int = 1):
    n, c, h, w = x.shape
    out_h, out_w = (h - kernel_size) // stride + 1, (w - kernel_size) // stride + 1

    output = np.zeros((n, out_h, out_w, c, kernel_size, kernel_size))
    for row_start in range(kernel_size):
        row_end = row_start + h - kernel_size
        for col_start in range(kernel_size):
            col_end = col_start + w - kernel_size

            patch = x[..., row_start : row_end + 1 : stride, col_start : col_end + 1 : stride]  # N, C, p1, p2
            output[..., row_start, col_start] = patch.transpose(0, 2, 3, 1)
    return output.reshape(n, out_h, out_w, -1)