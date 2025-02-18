#2025-02-12,13点39
# https://pywavelets.readthedocs.io/en/latest/

import numpy as np
import pywt
import cv2
import pywt
import cv2
import pywt
import cv2
from PIL import Image
import numpy as np
img=Image.open('feizhou.png')
img = img.convert('L')
see=np.array(img)
cA,(cH,cV,cD)=pywt.dwt2(img, 'haar')

cV[abs(cV)<9999]=0
cH[abs(cH)<9999]=0
cD[abs(cD)<9999]=0
print('原始图片')
print(see)
cv2.imwrite("rimgca.jpg",np.uint8(cA))
# 根据小波系数重构图像

rimg=pywt.idwt2((cA,(cH,cV,cD)),"haar")
print("-----------")
print('新图片')
print(rimg)
cv2.imwrite("rimg.jpg",np.uint8(rimg))








