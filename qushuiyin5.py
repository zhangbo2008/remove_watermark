# 继续研究去水印的方法.
import cv2
# a=cv2.imread('999.jpg')
a=cv2.imread('888.jpg')
hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)

# 分离通道
h, s, v = cv2.split(hsv)
cv2.imwrite('debugh.png',h)
print('尝试模糊s里面的值')
cv2.imwrite('debugs.png',s)

cv2.imwrite('debugv.png',v)

s[914:966,879:1013]=92
cv2.imwrite('debugs2.png',s)
v2=v[914:966,879:1013]
v2[v2<=250]=255

v[914:966,879:1013]=v2





h2=h[914:966,879:1013]
h2[h2<=30]=30

h[914:966,879:1013]=h2



cv2.imwrite('debugv2.png',v)


import numpy as np
#=局部亮度变低.
hsv = cv2.merge([h, s, v])

tmp2=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)



# tmp3=tmp2[1000:1450,300:2000]
# tmp3=np.uint8(np.float32(tmp3)*0.99//1)
# tmp2[1000:1450,300:2000]=tmp3



import cv2

# 合并通道


yuantu=cv2.imwrite('10000.jpg',tmp2)