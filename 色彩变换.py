#2025-02-11,15点35 
# 尝试用色彩变换来 去噪:https://zhuanlan.zhihu.com/p/95952096


import cv2
# a=cv2.imread('999.jpg')
a=cv2.imread('test.png')
hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)

fruit = cv2.imread("feizhou.png")
fruit = cv2.cvtColor(fruit,cv2.COLOR_BGR2YUV)
Y,U,V = cv2.split(fruit)

cv2.imwrite('1.png',Y)
cv2.imwrite('2.png',U)
cv2.imwrite('3.png',V)

