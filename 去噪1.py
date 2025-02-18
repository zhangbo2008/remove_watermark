# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
 
#读取图像
img = cv2.imread('feizhou.png', 0)
 
#傅里叶变换
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
dftshift = np.fft.fftshift(dft)
res1= 20*np.log(cv2.magnitude(dftshift[:,:,0], dftshift[:,:,1]))
 
 
 
 
 
# mask :
# mask:
threshold=10
rows, cols = dftshift.shape[:2]
crow, ccol = rows//2 , cols//2
# 创建一个和原图一样大小的掩码，用于低通滤波  #我们只保留图片中心部分.
mask = np.ones((rows, cols, 2), np.uint8)*1.0
mask[227,178]=0.0
dftshift=dftshift*mask
 
 
 
 
 
#傅里叶逆变换
ishift = np.fft.ifftshift(dftshift)
iimg = cv2.idft(ishift)
res2 = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])
 
#显示图像
plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(132), plt.imshow(res1, 'gray'), plt.title('Fourier Image')
plt.axis('off')
plt.subplot(133), plt.imshow(res2, 'gray'), plt.title('Inverse Fourier Image')
plt.axis('off')
plt.savefig('zzzzzzzzz.png')

#========figure进行图像重置.
plt.figure()
plt.imshow(img, 'gray')
plt.savefig('1.png')



plt.figure()
plt.imshow(res1, 'gray')
plt.savefig('2.png')

plt.figure()
plt.imshow(res2, 'gray')
plt.savefig('3.png')




