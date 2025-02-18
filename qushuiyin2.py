import cv2
a=cv2.imread('999.jpg')
a=cv2.imread('tmp222222.png')
hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)

# 分离通道
h, s, v = cv2.split(hsv)
cv2.imwrite('debugh.png',h)
cv2.imwrite('debugs.png',s)
cv2.imwrite('debugv.png',v)



# 对明度通道进行阈值处理
_, v = cv2.threshold(v, 180, 255, cv2.THRESH_BINARY) # 大于190的都变0.

# 合并通道
hsv = cv2.merge([h, s, v])

yuantu=cv2.imwrite('10000.jpg',cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))