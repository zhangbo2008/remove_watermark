# 继续研究去水印的方法.
import cv2
# a=cv2.imread('999.jpg')
a=cv2.imread('666.jpg')
hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)

# 分离通道
h, s, v = cv2.split(hsv)
cv2.imwrite('debugh.png',h)
print('尝试模糊s里面的值')


import  numpy as np
s[1000:1450,300:2000]=s[1000:1450,300:2000]//1

tmp=s[1000:1450,300:2000]
tmp[np.bitwise_and(tmp<80 , 50<tmp)]=73
s[1000:1450,300:2000]=tmp

#===处理小字
tmp=s[1229:1446,420:700]
tmp[np.bitwise_and(tmp<115 , 80<tmp)]=115
s[1229:1446,420:700]=tmp


#===处理胖字
tmp=s[1229:1446,700:1000]
tmp[np.bitwise_and(tmp<80 , 70<tmp)]=93
s[1229:1446,700:1000]=tmp


#===处理看字
tmp=s[1229:1446,1200:1900]
tmp[np.bitwise_and(tmp<80 , 70<tmp)]=93
s[1229:1446,1200:1900]=tmp

#处理房字
tmp=s[1229:1446,1200:1900]
tmp[np.bitwise_and(tmp<115 , 80<tmp)]=115
s[1229:1446,1200:1900]=tmp







cv2.imwrite('debugs.png',s)

cv2.imwrite('debugv.png',v)
#=========经过分离通道的分析, 我们的水印都在s和v这两个图里面.


# 对明度通道进行阈值处理
if 0:
  for yuzhi in range(150,250,5):
    _, v2 = cv2.threshold(v, yuzhi, 255, cv2.THRESH_BINARY) # 大于190的都变0.
    cv2.imwrite(f'debugv{yuzhi}.png',v2)
    
    
#================第一步去掉斜的水印小字:

v[v>237]=245   # 这个地方不写255写250效果更好.

    
    
print('180阈值挺好')
tmp=v[1000:1450,300:2000]
tmp[tmp>180]=250
v[1000:1450,300:2000]=tmp


#========v再细扣:


tmp=v[1156:1407,360:880]
tmp[np.bitwise_and(tmp>70 , 95>tmp)]=47
v[1156:1407,360:880]=tmp








cv2.imwrite('debugv.png',v)
# _, v = cv2.threshold(v, 190, 255, cv2.THRESH_BINARY) # 大于190的都变0.




#=局部亮度变低.
hsv = cv2.merge([h, s, v])

tmp2=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)



tmp3=tmp2[1000:1450,300:2000]
tmp3=np.uint8(np.float32(tmp3)*0.99//1)
tmp2[1000:1450,300:2000]=tmp3



import cv2

# 合并通道


yuantu=cv2.imwrite('10000.jpg',tmp2)