import json
import numpy as np
import os
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn import mixture
import pandas as pd
import cv2
from statistics import mean

height_list = [720, 720, 720, 720, 720, 480, 1920, 640]
width_list = [576, 576, 576, 576, 576, 360, 1080, 360]

# BGR
colors = [(0, 0, 255),
          (0, 255, 0),
          (255, 0, 0),
          (255, 255, 0),
          (255, 0, 255),
          (0, 255, 255)]

name_list = ['car_surveillance',
             'Zhongshan-East-cap',
             'Zhongshan-West-cap',
             'Zhongshan-South',
             'Zhongshan-North',
             'video1',
             'video3',
             'video6']

# Parameters
video_index = 2
weight = 0
my_cluster_num = 2
cv_types = ['spherical', 'tied', 'diag', 'full']
cv_type = cv_types[3]

video_name = name_list[video_index]
height = height_list[video_index]
width = width_list[video_index]

base_json = r'E:\MIT\Processed Data'
json_path = os.path.join(base_json, video_name + '.json')
base_cloud = os.path.join('./PointCloud', video_name)


def draw_point_cloud(vehicle_info_all):
    img = np.zeros((height, width, 3), np.uint8)
    for frame_num, frame_info in vehicle_info_all.items():
        for car_index, vehicle_info in frame_info.items():
            # overlay = img.copy()
            cv2.circle(img, (int(vehicle_info['cx']), int(vehicle_info['cy'])), 1, (0, 0, 255), -1)
            # img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.imwrite(os.path.join(base_cloud, video_name+'_source', ".png"), img)


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
    # clustering = DBSCAN(eps=0.5, min_samples=2).fit(vehicle_data)
    gmm = mixture.GaussianMixture(n_components=my_cluster_num, covariance_type=cv_type)

    # label_list = clustering.labels_
    label_list = gmm.fit_predict(vehicle_data)
    if min(label_list) < 0:
        label_list = label_list + (0-min(label_list))

    label_dict = {}
    for key in label_list:
        label_dict[key] = label_dict.get(key, 0) + 1
    print(label_dict)

    cluster_num = len(set(label_list))

    img_list = []
    img = np.zeros((height, width, 3), np.uint8)
    for i in range(cluster_num):
        img_list.append(np.zeros((height, width, 3), np.uint8))

    for i, l in enumerate(label_list):
        cv2.circle(img, (int(cx[i]), int(cy[i])), 1, colors[l], -1)
        cv2.circle(img_list[l], (int(cx[i]), int(cy[i])), 1, colors[l], -1)

    # Save cluster info
    save_cluster_info(cluster_num, df, label_list)

    cv2.imwrite(os.path.join(base_cloud, video_name+'_cluster', ".png"), img)
    for i in range(cluster_num):
        tmp_path = os.path.join(base_cloud, video_name + '_' + str(i) + '.png')
        cv2.imwrite(tmp_path, img_list[i])


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
    data_path = os.path.join(base_cloud, video_name.split('.')[0] + '.json')
    f = open(data_path, 'w')
    f.write(j)
    f.close()

 
def main():
    if not os.path.exists(base_cloud):
        os.mkdir(base_cloud)
    json_file = open(json_path)
    json_string = json_file.read()
    vehicle_info_all = json.loads(json_string)

    draw_point_cloud(vehicle_info_all)
    point_cloud_cluster(vehicle_info_all)


if __name__ == '__main__':
    main()
