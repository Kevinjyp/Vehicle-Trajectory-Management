import json
import numpy as np
import os
from sklearn.cluster import KMeans, SpectralClustering
import pandas as pd
import cv2
from statistics import mean

height_list = [720, 720, 480, 1920, 640]
width_list = [576, 576, 360, 1080, 360]

base = r'E:\MIT\Processed Data'
name_list = ['car_surveillnace', 'LOOP', 'video1', 'video3', 'video6']

# Parameters
i = 1
weight = 50
cluster_num = 3

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

    for frame_index, frame_info in vehicle_info_all.items():
        for car_index, vehicle_info in frame_info.items():
            cx.append(vehicle_info['cx'])
            cy.append(vehicle_info['cy'])
            theta.append(vehicle_info['theta'] * weight)

    df = pd.DataFrame(list(zip(cx, cy, theta)), columns=['cx', 'cy', 'theta'])

    theta_min, theta_max = 0, 0
    for index, row in df.iterrows():
        vehicle_data.append([row['cx'], row['cy'], row['theta']])

    # Cluster Method
    clustering = KMeans(n_clusters=cluster_num, random_state=0).fit(vehicle_data)
    # clustering = SpectralClustering(n_clusters=cluster_num, assign_labels='discretize', random_state=0).fit(vehicle_data)

    print(clustering.labels_)
    print(min(clustering.labels_), max(clustering.labels_))

    img = np.zeros((height, width, 3), np.uint8)

    # BGR
    colors = [(0, 0, 255),
              (0, 255, 0),
              (255, 0, 0),
              (255, 255, 0),
              (255, 0, 255),
              (0, 255, 255)]

    for i, l in enumerate(clustering.labels_):
        cv2.circle(img, (int(cx[i]), int(cy[i])), 1, colors[l], -1)

    cluster_theta = get_cluster_theta(cluster_num, df, clustering.labels_)

    for label in range(cluster_num):
        cluster_theta[str(label)]['color'] = colors[label]

    # Save Vehicle Position and Direction Information
    j = json.dumps(cluster_theta, indent=1)
    data_path = os.path.join(r'G:\留学\MIT暑研\资料\Code\Trajectory\cloud point\cluster', video_name.split('.')[0] + '.json')
    f = open(data_path, 'w')
    f.write(j)
    f.close()

    cv2.imwrite(cluster_point_path, img)


def get_cluster_theta(n_cluster, point_df, cluster_label):
    all_cluster_theta = {}
    cluster_init = [False for i in range(n_cluster)]
    for index, row in point_df.iterrows():
        if not cluster_init[cluster_label[index]]:
            cluster_init[cluster_label[index]] = True
            all_cluster_theta[cluster_label[index]] = []
        all_cluster_theta[cluster_label[index]].append(row['theta']/weight)

    return_list = {}
    for key, value in all_cluster_theta.items():
        return_list[str(key)] = {}
        return_list[str(key)]['min'] = min(value)
        return_list[str(key)]['max'] = max(value)
        return_list[str(key)]['avg'] = mean(value)
        return_list[str(key)]['point_num'] = len(value)

    return return_list


def main():
    json_file = open(json_path)
    json_string = json_file.read()
    vehicle_info_all = json.loads(json_string)

    draw_point_cloud(vehicle_info_all)
    point_cloud_cluster(vehicle_info_all)


if __name__ == '__main__':
    main()
