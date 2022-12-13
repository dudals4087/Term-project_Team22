import cv2
import numpy as np

image = cv2.imread('image/sample.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
template = cv2.imread('image/template.jpg', 0)
w, h = template.shape[::-1]

result = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.89
box_loc = np.where(result >= threshold)

for box in zip(*box_loc[::-1]):
    startX, startY = box
    endX, endY = startX + w, startY + h
    cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 1)

cv2.imwrite('result.png', image)
