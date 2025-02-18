import cv2
a=cv2.imread('tmp3.jpg',0)
a[a>100]=255
cv2.imwrite('fffffffff.png',a)