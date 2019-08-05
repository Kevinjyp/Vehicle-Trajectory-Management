import json
import numpy as np
import cv2

height = 720
width = 576

# jsonFile = open(r"E:\MIT\Processed Data\car_surveillnace.json")
jsonFile = open(r"E:\MIT\Processed Data\LOOP.json")

jsonString = jsonFile.read()
vehicle_info_all = json.loads(jsonString)

img = np.zeros((height, width, 3), np.uint8)

alpha = 0.99  # Transparency factor.

for frame_num, frame_info in vehicle_info_all.items():
    for car_index, vehicle_info in frame_info.items():
        overlay = img.copy()
        cv2.circle(img, (int(vehicle_info['cx']), int(vehicle_info['cy'])), 1, (0, 0, 255), -1)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

cv2.imwrite("./point_cloud_1_6.png", img)

i = 0