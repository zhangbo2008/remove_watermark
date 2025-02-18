'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the image in gray scale
img = cv2.imread('feizhou.png',0)
rows, cols = img.shape

# Transform the image to improve the speed in the fourier transform calculation
optimalRows = cv2.getOptimalDFTSize(rows)
optimalCols = cv2.getOptimalDFTSize(cols)
optimalImg = np.zeros((optimalRows, optimalCols))
optimalImg[:rows, :cols] = img
crow, ccol = optimalRows // 2 , optimalCols // 2

# Calculate the discrete Fourier transform
dft = cv2.dft(np.float32(optimalImg), flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(dft)

# Mask everything except the center
mask = np.zeros((optimalRows, optimalCols, 2), np.uint8)
mask[crow - 50:crow + 50, :] = 1
dftShift = dftShift * mask

# Rescale the values for visualization purposes
magnitudeSpectrum = 20 * np.log(cv2.magnitude(dftShift[:, :, 0], dftShift[:, :, 1])+0.001)

# Reconstruct the image using the inverse Fourier transform
newDft = np.fft.ifftshift(dftShift)
result = cv2.idft(newDft)
result = cv2.magnitude(result[:, :, 0], result[:, :, 1])

# Display the results
images = [optimalImg, magnitudeSpectrum, result]
imageTitles = ['Input image', 'Magnitude Spectrum', 'Result']

for i in range(len(images)):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(imageTitles[i])
    plt.xticks([])
    plt.yticks([])

plt.savefig('out.png')

plt.figure()
plt.imshow(optimalImg, 'gray')
plt.savefig('1.png')



plt.figure()
plt.imshow(magnitudeSpectrum, 'gray')
plt.savefig('2.png')

plt.figure()
plt.imshow(result, 'gray')
plt.savefig('3.png')  # 不能用cv2来存有bug

