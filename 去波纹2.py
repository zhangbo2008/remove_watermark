import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('feizhou.png', cv2.IMREAD_GRAYSCALE)
from PIL import Image, ImageFilter
 
# 读取图片
image = Image.open('feizhou.png')
 
# 使用高斯模糊去除波纹（类似于OpenCV）
blurred_image = image.filter(ImageFilter.GaussianBlur(radius=5))
 
# 显示原图和模糊后的图片
image.show()
blurred_image.save("dfadsfadsfa.png")