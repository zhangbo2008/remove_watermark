import cv2
# opencv-contrib 
# 探测二维码
detect_obj = cv2.wechat_qrcode_WeChatQRCode('detect.prototxt', 'detect.caffemodel', 'sr.prototxt', 'sr.caffemodel')
img = cv2.imread('123.jpg')
img = cv2.imread('335.png')
res, points = detect_obj.detectAndDecode(img)

# 绘制框线
for pos in points:
    color = (0, 0, 255)
    thick = 3
    for p in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        start = int(pos[p[0]][0]), int(pos[p[0]][1])
        end = int(pos[p[1]][0]), int(pos[p[1]][1])
        cv2.line(img, start, end, color, thick)

cv2.imwrite('wechat-qrcode-detect.jpg', img)