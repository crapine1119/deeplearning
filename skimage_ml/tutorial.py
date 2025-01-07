import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, exposure, data, filters

raw_img = data.astronaut()

img = cv.cvtColor(raw_img, cv.COLOR_RGB2BGR)
plt.imshow(img)


# 로컬 히스토그램 평활화 적용
equalized_img = exposure.equalize_adapthist(img, clip_limit=0.03)

# 원본 이미지와 결과 이미지 표시
io.imshow_collection([img, equalized_img])
io.show()
plt.figure()
plt.imshow(filters.gaussian(img, sigma=20))
##
from skimage.color import rgb2hsv
from skimage.transform import resize, rescale


plt.figure()
plt.imshow(resize(rgb2hsv(raw_img), (1024, 512)))


plt.figure()
plt.imshow(rescale(raw_img, scale=0.5, channel_axis=-1))

gray_img = cv.cvtColor(raw_img, cv.COLOR_RGB2GRAY)
hist = cv.calcHist(gray_img, [0], None, [256], [0, 256])
plt.plot(hist)

##

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity

# 이미지 불러오기 (BGR 형식)
image = data.astronaut()

x_pos = np.array([list(range(image.shape[0]))] * image.shape[1])
y_pos = x_pos.T

# 이미지 데이터를 2D 배열로 변환
pixels = image.reshape(-1, 3)
pixel_with_pos = np.dstack([image, np.expand_dims(x_pos, -1), np.expand_dims(y_pos, -1)])
pixel_with_pos = pixel_with_pos.reshape(-1, 5)

# K-means 클러스터링 적용
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
# kmeans.fit(pixels)
kmeans.fit(pixel_with_pos)

# 클러스터링 결과를 바탕으로 새로운 이미지 생성
segmented_pixels = kmeans.cluster_centers_[kmeans.labels_].astype("uint8")
# segmented_image = segmented_pixels.reshape(image.shape)
segmented_image = segmented_pixels[..., :3].reshape(image.shape)

# 원본 이미지와 분할된 이미지 비교
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(segmented_image)
axes[1].set_title(f"Segmented Image (K={k})")
axes[1].axis("off")

plt.show()

structural_similarity(image, segmented_image, channel_axis=2)

filterd_image = (filters.gaussian(image, sigma=20) * 255).astype(int)
structural_similarity(image, filterd_image, channel_axis=2)
