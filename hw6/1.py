import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# 使用canny算法检测边缘
def process(image):
    canny = cv2.Canny(image, 150, 400)

    # 使用霍夫变换检测直线
    # lines = cv2.HoughLinesP(canny, rho=1, theta=np.pi/180, threshold=150, minLineLength=50, maxLineGap=50)

    # 自己实现霍夫变换检测直线
    h, w = canny.shape
    max_rho = int(np.sqrt(h ** 2 + w ** 2))
    thetas = np.deg2rad(np.arange(-90, 90))
    rhos = np.linspace(-max_rho, max_rho, 2 * max_rho)

    accumulator = np.zeros((2 * max_rho, len(thetas)), dtype=np.int32)
    y_idxs, x_idxs = np.nonzero(canny)  # 获取边缘点的坐标

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(len(thetas)):
            rho = int(x * np.cos(thetas[t_idx]) + y * np.sin(thetas[t_idx]) + max_rho)
            accumulator[rho, t_idx] += 1

    threshold = 150  # 阈值
    lines = []
    for rho in range(accumulator.shape[0]):
        for theta in range(accumulator.shape[1]):
            if accumulator[rho, theta] > threshold:
                rho_val = rhos[rho]
                theta_val = thetas[theta]
                a = np.cos(theta_val)
                b = np.sin(theta_val)
                x0 = a * rho_val
                y0 = b * rho_val
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                lines.append((x1, y1, x2, y2))

    # 在原始图像上绘制直线
    result = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return canny, result

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

image_path = r"pic\\Fig1034(a)(marion_airport).tif"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 处理图像
canny, result = process(image)

# 保存处理后的图像
cv2.imwrite("result\\1-1.jpg", canny)
cv2.imwrite("result\\1-2.jpg", result)