import cv2
import numpy as np

image = cv2.imread('./image/example1.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
template = cv2.imread('./image/example2.jpg', 0)
w, h = template.shape[::-1]

result = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

startX, startY = max_loc # 만약 cv.TM_SQDIFF 혹은 cv.TM_SQDIFF_NORMED를 사용했을경우 최솟값을 사용해야한다.
endX, endY = startX + w, startY + h
cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 5)

cv2.imwrite('result.png', image)