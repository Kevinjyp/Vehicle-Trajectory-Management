import cv2
import numpy as np

path = r'E:\MIT\Picture\erosion.png'

import cv2

image = cv2.imread('./point_cloud.png')
overlay = image.copy()

x, y, w, h = 10, 10, 10, 10  # Rectangle parameters
cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 200, 0), -1)  # A filled rectangle

alpha = 0.4  # Transparency factor.

# Following line overlays transparent rectangle over the image
image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

cv2.imshow("image", image)
cv2.imshow("overlay", overlay)
cv2.imshow("image_new", overlay)

