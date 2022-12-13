import cv2
import numpy as np

img = cv2.imread('image/board.jpg')
image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
example = cv2.imread('image/example.jpg', 0)
w, h = example.shape[::-1]

result = cv2.matchTemplate(image_gray, example, cv2.TM_CCOEFF_NORMED)

minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

Xdot, Ydot = maxLoc
toX, toY = Xdot + w, Ydot + h
cv2.rectangle(img, (Xdot, Ydot), (toX, toY), (0, 0, 255), 1)

cv2.imwrite('result.png', img)
