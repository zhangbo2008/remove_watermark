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
mask=mask<100
img=img*mask
if 0:
 for i in range(kuandu,len(img)-kuandu):
   for j in range(kuandu,len(img[0])-kuandu):
      if mask[i][j]:#在附近找一个亮的涂上.
         for  d in candidata_index:
              if  not  mask[i+d[0],j+d[1]] :
                 
                img[i][j]=img[i+d[0],j+d[1]]
        


cv2.imwrite('12312312312312312312.png',img)





