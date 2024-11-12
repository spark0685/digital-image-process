import imageio
import numpy as np


# 投影变换
def transform(image):
    size = image.shape
    #x为角度，y为图片最大直径，根据图片大小创建一个新的图片
    round_size = int(np.sqrt(size[0]**2 + size[1]**2))
    result = np.zeros((180, round_size), dtype=int)

    img_sum = image.copy()
    for i in range(1, size[0]):
        img_sum[i] += img_sum[i-1]


    for angle_ in range(180):
        print(f'angle: {angle_}')
        for round in range(-round_size//2, round_size//2 + 1):
            #if np.abs(angle_ - 90) > 45:
                angle = np.deg2rad(angle_)
                x = np.arange(size[0])
                y_ = -x + size[0] / 2

                x_ = round / np.cos(angle) - y_ * np.tan(angle)  # 结果是一个 (size[0], round_size) 的数组
                y = (x_ + size[1] / 2).astype(int)

                valid_mask = (y >= 0) & (y < size[1])
                result[angle_][round + (round_size // 2) - 1] += np.square(np.cos(angle))*image[x[valid_mask], y[valid_mask]].sum()
            #else:
                angle = np.deg2rad(angle_)
                y = np.arange(size[1])
                x_ = (y - size[1] / 2).astype(int)
                y_ = round / np.sin(angle) - x_ / np.tan(angle)
                x = (-y_  + size[0] / 2).astype(int)
                valid_mask = (x >= 0) & (x < size[0])
                    
                result[angle_][round + (round_size // 2) - 1] += np.square(np.sin(angle))*image[x[valid_mask], y[valid_mask]].sum()

    return result
            


# 主函数
if __name__ == '__main__':    
    filename1 = 'Fig0539(a)(vertical_rectangle)'

    filename2 = 'Fig0539(c)(shepp-logan_phantom)'

    # 读取TIFF图像
    image1 = imageio.imread(f'{filename1}.tif')
    image2 = imageio.imread(f'{filename2}.tif')
    result1 = transform(image1)
    result2 = transform(image2)

    # 保存图片
    imageio.imwrite(f'{filename1}_transform.png', result1)
    imageio.imwrite(f'{filename2}_transform.png', result2)
