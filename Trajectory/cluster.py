import json
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
import pandas as pd
import cv2
import matplotlib.pyplot as plt


height = 720
width = 576

# jsonFile = open(r"E:\MIT\Processed Data\car_surveillnace.json")
jsonFile = open(r"E:\MIT\Processed Data\LOOP.json")

jsonString = jsonFile.read()
vehicle_info_all = json.loads(jsonString)

cx = []
cy = []
theta = []
vehicle_data = []
weight = 50
cluster_num = 6

for frame_index, frame_info in vehicle_info_all.items():
    for car_index, vehicle_info in frame_info.items():
        cx.append(vehicle_info['cx'])
        cy.append(vehicle_info['cy'])
        theta.append(vehicle_info['theta']*weight)

df = pd.DataFrame(list(zip(cx, cy, theta)), columns=['cx', 'cy', 'theta'])

for index, row in df.iterrows():
    vehicle_data.append([row['cx'], row['cy'], row['theta']])

# clustering = KMeans(n_clusters=cluster_num, random_state=0).fit(vehicle_data)
clustering = SpectralClustering(n_clusters=cluster_num, assign_labels="discretize",random_state=0).fit(vehicle_data)

print(clustering.labels_)
print(min(clustering.labels_), max(clustering.labels_))

img = np.zeros((height, width, 3), np.uint8)

colors = [(0,0,255), (0,255,0), (255,0,0), (255,255,0), (255,0,255), (0,255,255)]

for i, l in enumerate(clustering.labels_):
    cv2.circle(img, (int(cx[i]), int(cy[i])), 1, colors[l], -1)

cv2.imwrite("./point_cloud_1_6_cluster.png", img)

