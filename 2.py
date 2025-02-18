import cv2
import numpy as np



def Remove_watermark(image):
    hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_range = np.array([140, 100, 90])
    high_range = np.array([185, 255, 255])
    mask = cv2.inRange(hue_image, low_range, high_range)

    kernel = np.ones((3, 3), np.uint8)
    dilate_img = cv2.dilate(mask, kernel, iterations=1)
    res = cv2.inpaint(image,dilate_img,5,flags=cv2.INPAINT_TELEA)

    cv2.imwrite('tmp222222.png',res)

if __name__ == '__main__':
    image = cv2.imread('tmp.png')
    Remove_watermark(image)
