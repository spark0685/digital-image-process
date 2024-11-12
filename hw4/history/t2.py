import numpy as np

# 卷积函数，使用mode='full'
def convolve(signal1, signal2):
    # 获取输入信号的长度
    len_signal1 = len(signal1)
    len_signal2 = len(signal2)
    
    # 计算卷积结果的长度
    output_length = len_signal1 + len_signal2 - 1
    output = np.zeros(output_length)
    
    # 进行卷积操作
    for i in range(len_signal1):
        for j in range(len_signal2):
            output[i + j] += signal1[i] * signal2[j]
    
    return output

# 示例使用
if __name__ == "__main__":
    # 示例信号
    signal1 = np.array([1, 2, 3])
    signal2 = np.array([0, 1, 0.5])

    # 进行卷积
    result = convolve(signal1, signal2)

    print("卷积结果:", result)
