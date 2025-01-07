import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt
from skimage import data

img = data.astronaut()
img.shape
##
patch_size = 128

h_seq_len = img.shape[0] // patch_size
w_seq_len = img.shape[1] // patch_size

img_reshaped = img.reshape(h_seq_len, patch_size, w_seq_len, patch_size, -1)
img_reshaped = np.transpose(img_reshaped, (0, 2, 1, 3, 4))

fig = plt.figure()
axes = fig.subplots(h_seq_len, w_seq_len)

for enum, ax in enumerate(axes.flatten()):
    i, j = enum // w_seq_len, enum % w_seq_len
    ax.imshow(img_reshaped[i, j])
##


patchs = rearrange(img, "(h p1) (w p2) c -> (h w) p1 p2 c", p1=patch_size, p2=patch_size)

fig = plt.figure()
axes = fig.subplots(h_seq_len, w_seq_len)

for enum, ax in enumerate(axes.flatten()):
    ax.imshow(patchs[enum])
