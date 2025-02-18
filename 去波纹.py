import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('feizhou.png', cv2.IMREAD_GRAYSCALE)

# 小波变换
coeffs = pywt.wavedec2(image, 'haar', level=2)
cA, (cH, cV, cD),_ = coeffs

# 对高频系数进行阈值处理，以去除条纹
threshold = 0.1
cH[np.abs(cH) < threshold] = 0
cV[np.abs(cV) < threshold] = 0
cD[np.abs(cD) < threshold] = 0

# 重构图像
new_coeffs = (cA, (cH, cV, cD)),_
denoised_image = pywt.waverec2(new_coeffs, 'haar')

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('原始图像')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('去除条纹后的图像')
plt.imshow(denoised_image, cmap='gray')
plt.axis('off')

plt.savefig('人脸去噪4.png')