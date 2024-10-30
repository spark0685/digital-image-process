import cv2
import numpy as np
import os

from myHisteq import Histeq

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

filename = "bridge"
im = cv2.imread(f"{filename}.jpg", cv2.IMREAD_GRAYSCALE)
n_1 = 2
n_2 = 64
n_3 = 256

im_histeq_1 = Histeq(im, n_1)
im_histeq_2 = Histeq(im, n_2)
im_histeq_3 = Histeq(im, n_3)


cv2.imwrite(f"result/_{filename}_eq_{n_1}.jpg", im_histeq_1)
cv2.imwrite(f"result/_{filename}_eq_{n_2}.jpg", im_histeq_2)
cv2.imwrite(f"result/_{filename}_eq_{n_3}.jpg", im_histeq_3)

import matplotlib.pyplot as plt

plt.figure(1)
plt.subplot(221)
plt.imshow(im, cmap="gray")
plt.title("original image")
plt.axis("on")
plt.subplot(222)
plt.imshow(im_histeq_1, cmap="gray")
plt.title(f"n={n_1}")
plt.axis("on")
plt.subplot(223)
plt.imshow(im_histeq_2, cmap="gray")
plt.title(f"n={n_2}")
plt.axis("on")
plt.subplot(224)
plt.imshow(im_histeq_3, cmap="gray")
plt.title(f"n={n_3}")
plt.axis("on")
plt.show()
