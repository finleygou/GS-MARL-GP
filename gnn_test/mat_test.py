import numpy as np

# 创建一个三维数组
A = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# 原始数组的形状
print("Original shape:", A.shape)

# 转置数组，改变轴的顺序
B = A.transpose(1, 0, 2)

# 转置后数组的形状
# print("\nTransposed shape:", B.shape)
print("A is", A)

print("B is", B)