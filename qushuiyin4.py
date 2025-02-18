# 继续研究去水印的方法.
import cv2
# a=cv2.imread('999.jpg')
a=cv2.imread('feizhou.png')
hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)

# 分离通道
h, s, v = cv2.split(hsv)
cv2.imwrite('debugh.png',h)
if 0:
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


# # 对明度通道进行阈值处理
# if 0:
#   for yuzhi in range(150,250,5):
#     _, v2 = cv2.threshold(v, yuzhi, 255, cv2.THRESH_BINARY) # 大于190的都变0.
#     cv2.imwrite(f'debugv{yuzhi}.png',v2)
    
    
# #================第一步去掉斜的水印小字:

# v[v>237]=245   # 这个地方不写255写250效果更好.

    
    
# print('180阈值挺好')
# tmp=v[1000:1450,300:2000]
# tmp[tmp>180]=250
# v[1000:1450,300:2000]=tmp


# #========v再细扣:


# tmp=v[1156:1407,360:880]
# tmp[np.bitwise_and(tmp>70 , 95>tmp)]=47
# v[1156:1407,360:880]=tmp








# cv2.imwrite('debugv.png',v)
# # _, v = cv2.threshold(v, 190, 255, cv2.THRESH_BINARY) # 大于190的都变0.




# #=局部亮度变低.
# hsv = cv2.merge([h, s, v])

# tmp2=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)



# tmp3=tmp2[1000:1450,300:2000]
# tmp3=np.uint8(np.float32(tmp3)*0.99//1)
# tmp2[1000:1450,300:2000]=tmp3



# import cv2

# # 合并通道


# yuantu=cv2.imwrite('10000.jpg',tmp2)



print('重新写')

import cv2
import dlib
import cv2
 
# 加载预训练的人脸检测模型
detector = dlib.get_frontal_face_detector()
 
# 读取图片
img = a
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# 检测人脸
faces = detector(gray)
face=faces[0]
height=face.bottom()-face.top()
# 在检测到的人脸周围画矩形框
img2=img.copy()
if 1:
    x1, y1, x2, y2 = face.left(), int(face.top()-height*0.3), face.right(), int(face.bottom()+height*0.3)
    cv2.rectangle(img2, (x1, y1), (x2, y2), (255, 0, 0), 2)
 

cv2.imwrite('画人脸框.png', img2)


newpic=img[y1:y2,x1:x2]
cv2.imwrite('画人脸框2.png', newpic)









# #玩傅里叶: from :  https://blog.csdn.net/hellozhxy/article/details/132808742
# # -*- coding: utf-8 -*-
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
 
# #读取图像 # 只看第一个通道 # https://baijiahao.baidu.com/s?id=1759767004314074849&wfr=spider&for=pc
# img = newpic[:,:,0]
 
# #傅里叶变换
# # dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
# f = np.fft.fft2(img)
# dftshift = np.fft.fftshift(f)
# res1= 20*np.log(np.abs(dftshift))
 
# #傅里叶逆变换
# ishift = np.fft.ifftshift(dftshift)

# #显示图像
# plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
# plt.axis('off')
# plt.subplot(132), plt.imshow(res1, 'gray'), plt.title('Fourier Image')
# plt.axis('off')

# plt.axis('off')
# plt.axis('off')
# plt.savefig('人脸去噪1.png')










# #=========dierge 


# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
 
# def fft_image(image):
#     dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
#     dft_shift = np.fft.fftshift(dft)
#     magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
#     cv2.imwrite('fffff.png',magnitude_spectrum)
#     return dft_shift, magnitude_spectrum
 
# def ifft_image(dft_shift):
#     dft_ishift = np.fft.ifftshift(dft_shift)

#     image_back = cv2.idft(dft_ishift)
#     image_back = cv2.magnitude(dft_ishift[:,:,0], dft_ishift[:,:,1])

#     image_back = cv2.normalize(image_back, None, 0, 255, cv2.NORM_MINMAX)
#     image_back = np.abs(image_back)
#     cv2.imwrite('fffff2.png',image_back)
#     return image_back
 
# def denoise_image(image, threshold=1):
#     dft_shift, magnitude_spectrum = fft_image(image)
#     rows, cols = image.shape
#     crow, ccol = rows//2 , cols//2
#     # 创建一个和原图一样大小的掩码，用于低通滤波
#     mask = np.ones((rows, cols, 2), np.uint8)
#     mask[crow-threshold:crow+threshold, ccol-threshold:ccol+threshold] = 0
#     f = dft_shift
#     denoised_image = ifft_image(f)
#     return denoised_image
 
# # 读取图像并转为灰度图
# image =newpic[:,:,0]
# # 去噪处理
# denoised_image = denoise_image(image)
# # 显示结果
# plt.subplot(131),plt.imshow(image, cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(132),plt.imshow(denoised_image, cmap = 'gray')
# plt.title('Denoised Image'), plt.xticks([]), plt.yticks([])
# plt.savefig('人脸去噪2.png')









# #==
# img = newpic[:,:,0]
# dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
# dftshift = np.fft.fftshift(dft)
# res1= 20*np.log(cv2.magnitude(dftshift[:,:,0], dftshift[:,:,1]))
 
# #傅里叶逆变换
# ishift = np.fft.ifftshift(dftshift)
# iimg = cv2.idft(ishift)
# res2 = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])
# cv2.imwrite('3333333.png',res2)









# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
 
#读取图像
img = newpic[:,:,0]
 
#傅里叶变换
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
dftshift = np.fft.fftshift(dft)

res10= 20*np.log(cv2.magnitude(dftshift[:,:,0], dftshift[:,:,1]))
 
#傅里叶逆变换



# mask:
threshold=10
rows, cols = dftshift.shape[:2]
crow, ccol = rows//2 , cols//2
# 创建一个和原图一样大小的掩码，用于低通滤波  #我们只保留图片中心部分.
mask = np.ones((rows, cols, 2), np.uint8)*1.0

# mask[crow-threshold:crow+threshold, ccol-threshold:ccol+threshold,:] = 1
# dftshift[crow-threshold:crow+threshold, ccol-threshold:ccol+threshold,:]=0.1

dftshift=dftshift*mask

# dftshift 进行高斯模糊.


# 设置高斯核大小和标准差
sigma = 0.37

# 确定高斯核大小
kernel_size = int(6 * sigma + 1)  # 通常选择为 6*sigma + 1

# 使用 OpenCV 进行高斯模糊，方法1
a = cv2.GaussianBlur(dftshift, (0, 0), sigma)

# 使用 OpenCV 进行高斯模糊，方法2
kernel = cv2.getGaussianKernel(kernel_size, sigma)
kernel = kernel * kernel.transpose()

dftshift = cv2.filter2D(dftshift, -1, kernel)











res11= 20*np.log(cv2.magnitude(dftshift[:,:,0], dftshift[:,:,1])+0.00000000000001)
res1= 20*np.log(cv2.magnitude(dftshift[:,:,0], dftshift[:,:,1])+0.00000000000001)











ishift = np.fft.ifftshift(dftshift)
iimg = cv2.idft(ishift)
res2 = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])
# cv2.imwrite('sdfsadfasdfsdafasdfds.png',res2)
# cv2.imwrite('图像原始的傅里叶图.png',res10)
#显示图像
# plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
# plt.axis('off')
# plt.subplot(132), plt.imshow(res1, 'gray'), plt.title('Fourier Image')
# plt.axis('off')
# plt.subplot(133), plt.imshow(res2, 'gray'), plt.title('Inverse Fourier Image')
# plt.axis('off')
# plt.show()
#========figure进行图像重置.
plt.figure()
plt.imshow(img, 'gray')
plt.savefig('1.png')



plt.figure()
plt.imshow(res1, 'gray')
plt.savefig('2.png')

plt.figure()
plt.imshow(res2, 'gray')
plt.savefig('3.png')  # 不能用cv2来存有bug





# img = newpic[:,:,0]
# # img 
# img2=img.copy()
# kuandu=20
# for i in range(kuandu,len(img)):
#    for j in range(kuandu,len(img[0])):
#       if img[i][j]<140:
#         img2[i][j]=int(img[i-kuandu:i+kuandu,j-kuandu:j+kuandu].mean())
# img = newpic[:,:,1]
# # img 
# img3=img.copy()
# kuandu=20
# for i in range(kuandu,len(img)):
#    for j in range(kuandu,len(img[0])):
#       if img[i][j]<140:
#         img3[i][j]=int(img[i-kuandu:i+kuandu,j-kuandu:j+kuandu].mean())
        
# img = newpic[:,:,2]
# # img 
# img4=img.copy()
# kuandu=20
# for i in range(kuandu,len(img)):
#    for j in range(kuandu,len(img[0])):
#       if img[i][j]<140:
#         img4[i][j]=int(img[i-kuandu:i+kuandu,j-kuandu:j+kuandu].mean())  
        
# img5=np.concatenate([img2[:,:,np.newaxis],img3[:,:,np.newaxis],img4[:,:,np.newaxis]],axis=-1)
        
img4=newpic.copy()
kuandu=10
for i in range(kuandu,len(img)):
   for j in range(kuandu,len(img[0])):
      if img[i][j]<140:
        img4[i][j]=int(img[i-kuandu:i+kuandu,j-kuandu:j+kuandu].min())  
        
        
        
cv2.imwrite('人脸去噪4.png',img4)









# import pywt
# import numpy as np
# from skimage import io
# from skimage.restoration import denoise_wavelet
 
# # 读取图像
# image = io.imread('feizhou.png', as_gray=True)
 
# # 使用小波变换进行去噪
# threshold = 'soft'  # 可以选择'soft'或'hard'
# # keep(dictionary=False)  # 不保留变换字典
 
# # 使用内置函数进行去噪处理
# denoised_image = denoise_wavelet(image,   wavelet='haar')
 
# # 显示去噪后的图像
# # io.imshow(denoised_image)
# cv2.imwrite('人脸去噪5.png',denoised_image)




