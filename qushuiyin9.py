# 继续研究去水印的方法.
import cv2
# a=cv2.imread('999.jpg')
a=cv2.imread('feizhou.png')
hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)

# 分离通道
h, s, v = cv2.split(hsv)
cv2.imwrite('debugh.png',h)







cv2.imwrite('debugs.png',s)

cv2.imwrite('debugv.png',v)

import numpy as np
import cv2
from matplotlib import pyplot as plt
newpic=a
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


# # dftshift 进行高斯模糊.


# # 设置高斯核大小和标准差
# sigma = 0.37

# # 确定高斯核大小
# kernel_size = int(6 * sigma + 1)  # 通常选择为 6*sigma + 1

# # 使用 OpenCV 进行高斯模糊，方法1
# a = cv2.GaussianBlur(dftshift, (0, 0), sigma)

# # 使用 OpenCV 进行高斯模糊，方法2
# kernel = cv2.getGaussianKernel(kernel_size, sigma)
# kernel = kernel * kernel.transpose()

# dftshift = cv2.filter2D(dftshift, -1, kernel)











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




