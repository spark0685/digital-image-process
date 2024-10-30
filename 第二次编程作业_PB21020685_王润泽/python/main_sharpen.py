import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from mySharpen import Sharpen

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

filename = "moon.jpg"
im = cv2.imread(filename)
im_s = Sharpen(im)

cv2.imwrite("result/_moon_s.jpg", im_s)

plt.figure()
plt.subplot(121)
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.title("original image")
plt.axis("on")
plt.subplot(122)
plt.imshow(cv2.cvtColor(im_s, cv2.COLOR_BGR2RGB))
plt.title("sharpened image")
plt.axis("on")
plt.show()
