import json
import numpy as np
import os
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn import mixture
import pandas as pd
import cv2
from statistics import mean
import random


width_list = [720, 720, 720, 720, 720, 480, 1920, 640]
height_list = [576, 576, 576, 576, 576, 360, 1080, 360]

name_list = ['car_surveillance',
             'Zhongshan-East-cap',
             'Zhongshan-West-cap2_',  # 竖直直行
             'Zhongshan-West-cap3',  # 左右转弯
             'Zhongshan-West-cap4',  # 水平直行
             'Zhongshan-West',  # all 5000 frames
            ]
# Parameters
video_index = 5
weight = 50
my_cluster_num = 3
min_points_in_one_cluster = 100
first_cluster_method = 'kmeans'
twice_cluster = 1

cv_types = ['spherical', 'tied', 'diag', 'full']
cv_type = cv_types[1]

video_name = name_list[video_index]
height = 720
width = 576

base_json = r'E:\MIT\Processed Data'
base_point = os.path.join('./PointCloud', video_name)


def get_n_colors(n_colors):
    # BGR
    colors = [(0, 0, 255),
              (0, 255, 0),
              (255, 0, 0),
              (255, 255, 0),
              (255, 0, 255),
              (0, 255, 255)]
    return colors


def draw_point_cloud(vehicle_info_all, path):
    img = np.zeros((height, width, 3), np.uint8)
    for frame_num, frame_info in vehicle_info_all.items():
        for car_index, vehicle_info in frame_info.items():
            cv2.circle(img, (int(vehicle_info['cx']), int(vehicle_info['cy'])), 1, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(path, video_name + '_source' + ".png"), img)


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

    for index, row in df.iterrows():
        vehicle_data.append([row['cx'], row['cy'], row['theta']])

    # Clustering Method
    label_list, label_dict, cluster_num = my_cluster_method(vehicle_data, first_cluster_method)

    # Draw Points
    img, img_list = my_draw_points(label_list, label_dict, cx, cy, cluster_num)

    # Save cluster info
    if not weight == 0:
        save_cluster_info(cluster_num, df, label_list)

    # Save clustering result picture
    my_img_write(base_point, img, img_list)

    if twice_cluster == 1:
        # Twice Cluster
        vehicle_data_twice = [[], [], []]
        for index, row in df.iterrows():
            vehicle_data_twice[label_list[index]].append([row['cx'], row['cy']])

        for type_index, v_data in enumerate(vehicle_data_twice):
            # Clustering Method
            label_list_twice, label_dict_twice, cluster_num_twice = my_cluster_method(v_data, 'dbscan')

            cx_twice = []
            cy_twice = []
            for row in v_data:
                cx_twice.append(row[0])
                cy_twice.append(row[1])

            # Draw Points
            img_twice, img_list_twice = my_draw_points(label_list_twice, label_dict_twice, cx_twice, cy_twice, cluster_num_twice)

            # Save clustering result picture
            base_path_cluster = os.path.join(base_point, 'Twice' + str(type_index))
            if not os.path.exists(base_path_cluster):
                os.mkdir(base_path_cluster)
            my_img_write(base_path_cluster, img_twice, img_list_twice)

            img_source = np.zeros((height, width, 3), np.uint8)
            for row in v_data:
                cv2.circle(img_source, (int(row[0]), int(row[1])), 1, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(base_path_cluster, video_name + '_source' + ".png"), img_source)


def my_cluster_method(data, method):
    # Cluster Method
    if method == "kmeans":
        clustering = KMeans(n_clusters=my_cluster_num, random_state=0).fit(data)
    elif method == "dbscan":
        clustering = DBSCAN(eps=11, min_samples=15).fit(data)
    else:
        print("Wrong Method!")
        return 0, 0, 0
    # clustering = SpectralClustering(n_clusters=my_cluster_num, assign_labels='discretize', random_state=0).fit(vehicle_data)
    # clustering = DBSCAN(eps=11, min_samples=7).fit(vehicle_data)
    # gmm = mixture.GaussianMixture(n_components=my_cluster_num, covariance_type=cv_type)

    # Choose label type
    label_list = clustering.labels_
    # label_list = gmm.fit_predict(vehicle_data)

    label_dict = {}
    for key in label_list:
        label_dict[key] = label_dict.get(key, 0) + 1
    print(label_dict)

    # Ignore small clusters
    cluster_num = 0
    for key, values in label_dict.items():
        if judge_cluster(key, values):
            cluster_num = cluster_num + 1

    return label_list, label_dict, cluster_num


def my_draw_points(label_list, label_dict, cx, cy, cluster_num):
    # Init image
    img_list = []
    img = np.zeros((height, width, 3), np.uint8)
    for i in range(cluster_num):
        img_list.append(np.zeros((height, width, 3), np.uint8))

    colors = get_n_colors(cluster_num)
    color_flag = []
    # Draw points
    for i, l in enumerate(label_list):
        if judge_cluster(l, label_dict[l]):
            if l not in color_flag:
                color_flag.append(l)
            cv2.circle(img, (int(cx[i]), int(cy[i])), 1, colors[color_flag.index(l)], -1)
            cv2.circle(img_list[color_flag.index(l)], (int(cx[i]), int(cy[i])), 1, colors[color_flag.index(l)], -1)

    return img, img_list


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

    colors = get_n_colors(n_cluster)

    for label in range(n_cluster):
        theta_dict[str(label)]['color'] = colors[label]

    # Save Vehicle Position and Direction Information
    j = json.dumps(theta_dict, indent=1)
    data_path = os.path.join(base_point, video_name.split('.')[0] + '.json')
    f = open(data_path, 'w')
    f.write(j)
    f.close()


def main():

    json_file = open(os.path.join(base_json, video_name + '.json'))
    json_string = json_file.read()
    vehicle_info_all = json.loads(json_string)

    draw_point_cloud(vehicle_info_all, base_point)
    point_cloud_cluster(vehicle_info_all)


def my_img_write(path, img, img_list):
    if not os.path.exists(path):
        os.mkdir(path)
    file_list = [f for f in os.listdir(path) if f.endswith(".png")]
    for f in file_list:
        os.remove(os.path.join(path, f))

    cv2.imwrite(os.path.join(path, video_name + '_cluster' + ".png"), img)
    for i in range(len(img_list)):
        tmp_path = os.path.join(path, video_name + '_' + str(i) + '.png')
        cv2.imwrite(tmp_path, img_list[i])


if __name__ == '__main__':
    main()
