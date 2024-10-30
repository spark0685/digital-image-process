import cv2
import numpy as np
import os

import matplotlib.pyplot as plt

from myAverage import Average

from myMedian import Median

if not os.path.exists("result"):
    os.makedirs("result")

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)


filename = "circuit"
im = cv2.imread(f"{filename}.jpg")

kernel_size = 3
iterations = 20

im_a = Average(im, kernel_size, iterations)
im_m = Median(im, kernel_size, iterations)


cv2.imwrite(f"result/_{filename}_a.jpg", im_a)
cv2.imwrite(f"result/_{filename}_m.jpg", im_m)

# 显示原始图像
plt.subplot(131)  # 1行3列，第1个
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("on")

# 显示均值滤波图像
plt.subplot(132)  # 1行3列，第2个
plt.imshow(cv2.cvtColor(im_a, cv2.COLOR_BGR2RGB))
plt.title("Average Filter")
plt.axis("on")

# 显示中值滤波图像
plt.subplot(133)  # 1行3列，第3个
plt.imshow(cv2.cvtColor(im_m, cv2.COLOR_BGR2RGB))
plt.title("Median Filter")
plt.axis("on")
plt.show()
