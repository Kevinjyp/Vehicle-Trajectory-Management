import json
import numpy as np
import os
from sklearn.cluster import KMeans, SpectralClustering
import pandas as pd
import cv2

height_list = [720, 720, 480]
width_list = [576, 576, 360]

base = r'E:\MIT\Processed Data'
name_list = ['car_surveillnace', 'LOOP', 'video1']

# control which video to use
i = 1

video_name = name_list[i]
height = height_list[i]
width = width_list[i]

json_path = os.path.join(base, video_name + '.json')
cloud_point_path = os.path.join('./cloud point/source', video_name + '.png')
cluster_point_path = os.path.join('./cloud point/cluster', video_name + '.png')


def draw_point_cloud(vehicle_info_all):
    img = np.zeros((height, width, 3), np.uint8)
    for frame_num, frame_info in vehicle_info_all.items():
        for car_index, vehicle_info in frame_info.items():
            # overlay = img.copy()
            cv2.circle(img, (int(vehicle_info['cx']), int(vehicle_info['cy'])), 1, (0, 0, 255), -1)
            # img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.imwrite(cloud_point_path, img)


def point_cloud_cluster(vehicle_info_all):
    cx = []
    cy = []
    theta = []
    vehicle_data = []
    weight = 50
    cluster_num = 3

    for frame_index, frame_info in vehicle_info_all.items():
        for car_index, vehicle_info in frame_info.items():
            cx.append(vehicle_info['cx'])
            cy.append(vehicle_info['cy'])
            theta.append(vehicle_info['theta'] * weight)

    df = pd.DataFrame(list(zip(cx, cy, theta)), columns=['cx', 'cy', 'theta'])

    for index, row in df.iterrows():
        vehicle_data.append([row['cx'], row['cy'], row['theta']])

    # Cluster Method
    clustering = KMeans(n_clusters=cluster_num, random_state=0).fit(vehicle_data)
    # clustering = SpectralClustering(n_clusters=cluster_num, assign_labels='discretize', random_state=0).fit(vehicle_data)

    print(clustering.labels_)
    print(min(clustering.labels_), max(clustering.labels_))

    img = np.zeros((height, width, 3), np.uint8)

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for i, l in enumerate(clustering.labels_):
        cv2.circle(img, (int(cx[i]), int(cy[i])), 1, colors[l], -1)

    cv2.imwrite(cluster_point_path, img)


def main():
    json_file = open(json_path)
    json_string = json_file.read()
    vehicle_info_all = json.loads(json_string)

    draw_point_cloud(vehicle_info_all)
    point_cloud_cluster(vehicle_info_all)


if __name__ == '__main__':
    main()