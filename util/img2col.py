import numpy as np
import torch


def img2col(x: np.ndarray, kernel_size: int = 3, stride: int = 1) -> np.ndarray:
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


def img2col_tensor(x: torch.Tensor, kernel_size: int = 3, stride: int = 1) -> torch.Tensor:
    n, c, h, w = x.shape
    out_h, out_w = (h - kernel_size) // stride + 1, (w - kernel_size) // stride + 1

    output = torch.zeros((n, out_h, out_w, c, kernel_size, kernel_size))
    for row_start in range(kernel_size):
        row_end = row_start + h - kernel_size
        for col_start in range(kernel_size):
            col_end = col_start + w - kernel_size

            patch = x[..., row_start : row_end + 1 : stride, col_start : col_end + 1 : stride]  # N, C, p1, p2
            output[..., row_start, col_start] = patch.permute(0, 2, 3, 1)
    return output.reshape(n, out_h, out_w, -1)


if __name__ == "__main__":
    img = [[[[int(f"{k}{i}{j}") for i in range(1, 8)] for j in range(1, 8)] for k in range(1, 2)]]
    img = np.array(img)
    reshaped = img2col(img)
    reshaped.shape
    for i in range(9):
        print(reshaped[0, -1, -1, i])
