import json
import numpy as np
import os
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn import mixture
import pandas as pd
import cv2
from statistics import mean

width_list = [720, 720, 720, 720, 720, 480, 1920, 640]
height_list = [576, 576, 576, 576, 576, 360, 1080, 360]

# BGR
colors = [(0, 0, 255),
          (0, 255, 0),
          (255, 0, 0),
          (255, 255, 0),
          (255, 0, 255),
          (0, 255, 255)]

name_list = ['car_surveillance',
             'Zhongshan-East-cap',
             'Zhongshan-West-cap2',  # 竖直直行
             'Zhongshan-West-cap3',  # 左右转弯
             'Zhongshan-West-cap4',  # 水平直行
             'Zhongshan-South',
             'Zhongshan-North',
             'video1',
             'video3',
             'video6']

# Parameters
video_index = 2
weight = 0
my_cluster_num = 4
min_points_in_one_cluster = 100

cv_types = ['spherical', 'tied', 'diag', 'full']
cv_type = cv_types[1]

video_name = name_list[video_index]
height = height_list[video_index]
width = width_list[video_index]

base_json = r'E:\MIT\Processed Data'
base_point = os.path.join('./PointCloud', video_name)


def draw_point_cloud(vehicle_info_all):
    img = np.zeros((height, width, 3), np.uint8)
    for frame_num, frame_info in vehicle_info_all.items():
        for car_index, vehicle_info in frame_info.items():
            cv2.circle(img, (int(vehicle_info['cx']), int(vehicle_info['cy'])), 1, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(base_point, video_name + '_source' + ".png"), img)


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
    # clustering = KMeans(n_clusters=my_cluster_num, random_state=0).fit(vehicle_data)
    # clustering = SpectralClustering(n_clusters=my_cluster_num, assign_labels='discretize', random_state=0).fit(vehicle_data)
    clustering = DBSCAN(eps=10, min_samples=11).fit(vehicle_data)
    # gmm = mixture.GaussianMixture(n_components=my_cluster_num, covariance_type=cv_type)

    # Choose label type
    label_list = clustering.labels_
    # label_list = gmm.fit_predict(vehicle_data)

    label_dict = {}
    for key in label_list:
        label_dict[key] = label_dict.get(key, 0) + 1
    print(label_dict)

    cluster_num = max(label_list)+1

    # Init image
    img_list = []
    img = np.zeros((height, width, 3), np.uint8)
    for i in range(cluster_num):
        img_list.append(np.zeros((height, width, 3), np.uint8))

    # Draw points
    for i, l in enumerate(label_list):
        if judge_cluster(l, label_dict[l]):
            cv2.circle(img, (int(cx[i]), int(cy[i])), 1, colors[l], -1)
            cv2.circle(img_list[l], (int(cx[i]), int(cy[i])), 1, colors[l], -1)

    # Save cluster info
    if not weight == 0:
        save_cluster_info(cluster_num, df, label_list)

    # Save clustering result picture
    cv2.imwrite(os.path.join(base_point, video_name + '_cluster' + ".png"), img)
    for i in range(cluster_num):
        if judge_cluster(i, label_dict[i]):
            tmp_path = os.path.join(base_point, video_name + '_' + str(i) + '.png')
            cv2.imwrite(tmp_path, img_list[i])


def judge_cluster(key, number):
    if key < 0:
        return False
    if number < min_points_in_one_cluster:
        return False
    return True


def save_cluster_info(n_cluster, point_df, cluster_label):
    all_cluster_theta = {}
    cluster_init = [False for i in range(n_cluster)]
    for index, row in point_df.iterrows():
        if not cluster_init[cluster_label[index]]:
            cluster_init[cluster_label[index]] = True
            all_cluster_theta[cluster_label[index]] = []
        if not weight == 0:
            all_cluster_theta[cluster_label[index]].append(row['theta'] / weight)
        else:
            all_cluster_theta[cluster_label[index]].append(0)
    theta_dict = {}
    for key, value in all_cluster_theta.items():
        theta_dict[str(key)] = {}
        theta_dict[str(key)]['min'] = min(value)
        theta_dict[str(key)]['max'] = max(value)
        theta_dict[str(key)]['avg'] = mean(value)
        theta_dict[str(key)]['point_num'] = len(value)

    for label in range(n_cluster):
        theta_dict[str(label)]['color'] = colors[label]

    # Save Vehicle Position and Direction Information
    j = json.dumps(theta_dict, indent=1)
    data_path = os.path.join(base_point, video_name.split('.')[0] + '.json')
    f = open(data_path, 'w')
    f.write(j)
    f.close()


def main():
    if not os.path.exists(base_point):
        os.mkdir(base_point)
    file_list = [f for f in os.listdir(base_point) if f.endswith(".png")]
    for f in file_list:
        os.remove(os.path.join(base_point, f))

    json_file = open(os.path.join(base_json, video_name + '.json'))
    json_string = json_file.read()
    vehicle_info_all = json.loads(json_string)

    draw_point_cloud(vehicle_info_all)
    point_cloud_cluster(vehicle_info_all)


if __name__ == '__main__':
    main()
