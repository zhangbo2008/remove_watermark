import cv2
a=cv2.imread('999.jpg',0)
a[a>150]=255
# cv2.imwrite('100.png',a)
#=======把这个100.png作为mask,给原图做遮罩,youcan get colored waterremoved pic.
import numpy as np
#=====继续写完.
yuantu=cv2.imread('999.jpg',1)
dex=a>150
dex=dex[:,:,np.newaxis]
dex2=np.concatenate([dex,dex,dex],axis=2)
yuantu[dex2]=255

yuantu=cv2.imwrite('10000.jpg',yuantu)