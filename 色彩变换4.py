#2025-02-11,15点35 
# 尝试用色彩变换来 去噪:https://zhuanlan.zhihu.com/p/95952096


import cv2
# a=cv2.imread('999.jpg')
a=cv2.imread('test.png')
hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)

fruit = cv2.imread("feizhou.png")
# fruit = cv2.cvtColor(fruit,cv2.COLOR_BGR2YUV)
Y,U,V = cv2.split(fruit)

cv2.imwrite('1.png',Y)
cv2.imwrite('2.png',U)
cv2.imwrite('3.png',V)




# 处理: 做替换


demo=Y.copy()
img=demo
kuandu=10
gao=5

import numpy as np

out=[]
for i in range(-2,2):
   for j in range(-2,2):
      out.append([i,j])
out.sort(key=lambda x:x[0]**2+x[1]**2)
candidata_index=out


#==计算边界:
img_sobel2 = cv2.Sobel(img, -1, 1,0, ksize=3)
# img_sobel2[img_sobel2<50 & (img<100)]=0
# img_sobel=img_sobel2>150
mask=img_sobel2# true 代表波纹, false代表好地方.

kernel = np.ones((10, 10),np.uint8)
mask=cv2.dilate(mask,kernel)

cv2.imwrite('img_sobel2.png',img_sobel2)





#=====step1:计算sobel傅里叶:

#读取图像
img = img_sobel2
from matplotlib import pyplot as plt
#傅里叶变换
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
dftshift = np.fft.fftshift(dft)

res10= 20*np.log(cv2.magnitude(dftshift[:,:,0], dftshift[:,:,1]))
 

plt.imshow(res10, 'gray')
plt.savefig('2.png')





#step2:原图傅里叶

img=Y.copy()
from matplotlib import pyplot as plt
#傅里叶变换
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
dftshift2 = np.fft.fftshift(dft)

res101= 20*np.log(cv2.magnitude(dftshift2[:,:,0], dftshift2[:,:,1]))
 

plt.imshow(res101, 'gray')
plt.savefig('3.png')


#==setp3:变回去
dftshift3=0.7*dftshift2-dftshift*1.2


ishift = np.fft.ifftshift(dftshift3)
iimg = cv2.idft(ishift)
res2 = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])*100



plt.imshow(res2, 'gray')
plt.savefig('4.png')  # 不能用cv2来存有bug














