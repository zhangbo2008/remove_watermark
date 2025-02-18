#2025-02-11,15点35 
# 尝试用色彩变换来 去噪:https://zhuanlan.zhihu.com/p/95952096


import cv2


import numpy as np
import matplotlib.pyplot as plt
from pywt import Wavelet, dwt2, idwt2
import pywt

img=cv2.imread('feizhou.png',0)
img=img.astype(np.float32)
coeffs=pywt.dwt2(img,'haar')
cA,(cH,cV,cD)=coeffs
aaa=pywt.idwt2([cA,(cH,cV,cD)],'haar')

plt.figure()
plt.imshow(aaa)
plt.savefig('before99999999999999999999aaaaaaa9.png')








cA,(cH,cV,cD)=coeffs
plt.figure()
plt.subplot(221),plt.imshow(cA,'gray'),plt.title('A')
plt.subplot(222),plt.imshow(cH,'gray'),plt.title('H')
plt.subplot(223),plt.imshow(cV,'gray'),plt.title('V')
plt.subplot(224),plt.imshow(cD,'gray'),plt.title('D')
plt.savefig('999999999999999999999.png')

threshold = 100
cH[np.abs(cH) < threshold] = 0
cV[np.abs(cV) < threshold] = 0
cD[np.abs(cD) < threshold] = 0


plt.figure()
plt.subplot(221),plt.imshow(cA,'gray'),plt.title('A')
plt.subplot(222),plt.imshow(cH,'gray'),plt.title('H')
plt.subplot(223),plt.imshow(cV,'gray'),plt.title('V')
plt.subplot(224),plt.imshow(cD,'gray'),plt.title('D')
plt.savefig('91919191919.png')






aaa=pywt.idwt2([cA,(cH,cV,cD)],'haar')

plt.figure()
plt.imshow(aaa)
plt.savefig('99999999999999999999aaaaaaa9.png')
