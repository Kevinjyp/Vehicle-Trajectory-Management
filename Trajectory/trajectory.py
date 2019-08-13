import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import json
import os

base_path = 'E:/MIT'
video_list = ['car_surveillance.avi',
              'Zhongshan-East-cap.mkv',
              'Zhongshan-West-cap2.mkv',  # 竖直直行
              'Zhongshan-West-cap3.mkv',  # 左右转弯
              'Zhongshan-West-cap4.mkv',  # 水平直行
              'Zhongshan-West.ts',  # all 5000 frames
              ]
video_name = video_list[5]

video_path = os.path.join(base_path, 'Video', video_name)

frame_num = 0
current_frame = 0
previous_frame = 0
preprevious_frame = 0
background_frame = 0
gray_sum = 0
gray_sum_list_left_lane = []
vehicle_info_all = {}

# Params
low_area = 1800
high_area = 10000
frame_limit = 5000


def main():
    global frame_num, previous_frame, preprevious_frame, current_frame, frame_limit

    # while(True):
    cap = cv2.VideoCapture(video_path)
    #     gray_sum_list = []

    if not cap.isOpened():
        print('Error opening video stream')

    fgbg = cv2.createBackgroundSubtractorMOG2(100, 36.0, True)

    while cap.isOpened():

        # Capture frame-by-frame
        ret, source_img = cap.read()
        # Median blur
        source_img = cv2.medianBlur(source_img, 5)
        frame_num = frame_num + 1
        print(frame_num)

        if ret:
            if frame_num == 1:
                preprevious_frame = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
            if frame_num == 2:
                previous_frame = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
            if frame_num > 2:
                current_frame = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

                # Get Foreground Image
                # binary_foregourd_img = my_cdi()
                binary_foregourd_img = my_gmm(source_img, fgbg)

                # Frame Update
                preprevious_frame = previous_frame
                previous_frame = current_frame

                # vehicle counting
                # gray_sum = 0
                # for y in range(300, 390):
                #     gray_sum += binary_foregourd_img[340, y]
                # gray_sum_list_left_lane.append(gray_sum)

                # Opening, Closing, Erotion, Dilation
                morphology_img = my_morphology(binary_foregourd_img)

                # Decide which Picture to Use
                processed_img = morphology_img
                # processed_img = binary_foregourd_img

                # Find contours
                contours, _ = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # Contour check
                contours = check_contour(contours)

                # Draw Contours
                source_img = draw_contours(source_img, contours)

                # Get Moment Information of Image from contours
                vehicle_info = get_img_moment(contours)

                # Draw Central Point and Theta
                source_img = draw_theta(source_img, vehicle_info)

                # Image Show
                frame_list = [source_img, processed_img]
                name_list = ['source_img', 'processed_img']
                img_show_list(name_list, frame_list)

                # Save Theta on the Picture
                base_path_pic = processed_path = os.path.join(base_path, 'Picture', video_name)
                processed_path = os.path.join(base_path_pic, 'Processed')
                source_path = os.path.join(base_path_pic, 'Source')
                if not os.path.exists(base_path_pic):
                    os.mkdir(base_path_pic)
                if not os.path.exists(processed_path):
                    os.mkdir(processed_path)
                if not os.path.exists(source_path):
                    os.mkdir(source_path)
                my_save_picture(processed_path, processed_img)
                my_save_picture(source_path, source_img)

        if (cv2.waitKey(1) & 0xff == ord('q')) | frame_num >= frame_limit:
            break

    # Save Vehicle Position and Direction Information
    j = json.dumps(vehicle_info_all, indent=1)
    data_path = os.path.join(base_path, 'Processed Data', video_name.split('.')[0] + '.json')
    f = open(data_path, 'w')
    f.write(j)
    f.close()

    cap.release()


def my_save_picture(path, img):
    global frame_num
    if not os.path.exists(path):
        os.mkdir(path)
    cv2.imwrite(os.path.join(path, str(frame_num) + '.jpg'), img)


def my_gmm(frame, fgbg):
    # Background Learning
    fgmask = fgbg.apply(frame, learningRate=0.2)

    # Threshold
    _, fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)
    # fgmask = cv2.adaptiveThreshold(fgmask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return fgmask


def my_cdi():
    global preprevious_frame, previous_frame, current_frame
    dif1 = cv2.absdiff(current_frame, previous_frame)
    dif2 = cv2.absdiff(preprevious_frame, previous_frame)

    _, thresh1 = cv2.threshold(dif1, 20, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(dif2, 20, 255, cv2.THRESH_BINARY)

    binary_dif_cdi = cv2.bitwise_and(thresh1, thresh2)  # 与运算

    return binary_dif_cdi


def my_morphology(binary_foregourd_img):
    # Morphological Operation
    morphology_img = binary_foregourd_img
    # Close small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    morphology_img = cv2.morphologyEx(morphology_img, cv2.MORPH_CLOSE, kernel)
    # De-noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphology_img = cv2.morphologyEx(morphology_img, cv2.MORPH_OPEN, kernel)
    # # Dilation
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # morphology_img = cv2.erode(morphology_img, kernel, iterations=1)

    return morphology_img


def get_img_moment(contours):
    global frame_num, vehicle_info_all
    # Get Moment Information
    vehicle_info_frame = {}
    car_num = 0

    for cnt in contours:
        vehicle_info = {}
        m = cv2.moments(cnt)

        vehicle_info['cx'] = m['m10'] / m['m00']
        vehicle_info['cy'] = m['m01'] / m['m00']

        mu11_prime = m['mu11'] / m['m00']
        mu20_prime = m['mu20'] / m['m00']
        mu02_prime = m['mu02'] / m['m00']

        try:
            vehicle_info['theta'] = 0.5 * math.degrees(math.atan(2 * mu11_prime / (mu20_prime - mu02_prime)))
            vehicle_info['frame'] = frame_num
            vehicle_info_frame[car_num] = vehicle_info
            car_num += 1
        except ZeroDivisionError:
            err = str(vehicle_info['cx']) + ', ' + str(vehicle_info['cy']) + ', ' + str(frame_num) + '\n'
            err_path = os.path.join(base_path, 'Processed Data', video_name.split('.')[0] + '.txt')
            with open(err_path, 'a+') as f:
                f.write(err)

    vehicle_info_all[frame_num] = vehicle_info_frame
    return vehicle_info_frame


def img_show_list(name_list, frame_list):
    for name, frame in zip(name_list, frame_list):
        cv2.imshow(name, frame)


def draw_contours(source_img, contours):
    # rect contours
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(source_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return source_img


def draw_theta(source_img, vehicle_info):
    # Draw Central Point and Theta
    for key, value in vehicle_info.items():
        center = (int(vehicle_info[key]['cx']), int(vehicle_info[key]['cy']))
        cv2.circle(source_img, center, 2, (0, 0, 255))
        height = 50 * math.tan(math.radians(vehicle_info[key]['theta']))
        start = (int(vehicle_info[key]['cx'] - 25), int(vehicle_info[key]['cy'] + height))
        end = (int(vehicle_info[key]['cx'] + 25), int(vehicle_info[key]['cy'] - height))
        cv2.line(source_img, start, end, (0, 255, 0), thickness=3)
    return source_img


def check_contour(contours):
    global low_area, high_area

    new_contours = []
    flag = [True for i in range((len(contours)))]

    for idx, cnt in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(cnt)
        if not judge_contour_area(w, h, low_area, high_area):
            flag[idx] = False
        # For bicycles and people
        if h > 1.2 * w:
            flag[idx] = False

    # Discard nesting bounding box
    for idx_1, cnt_1 in enumerate(contours):
        if not flag[idx_1]:
            continue
        (x_1, y_1, w_1, h_1) = cv2.boundingRect(cnt_1)
        for idx_2, cnt_2 in enumerate(contours):
            if not flag[idx_2]:
                continue
            (x_2, y_2, w_2, h_2) = cv2.boundingRect(cnt_2)
            cx_2 = x_2 + w_2 / 2
            cy_2 = y_2 + h_2 / 2
            if x_1 < cx_2 < x_1 + w_1 and y_1 < cy_2 < y_1 + h_1 and w_2*h_2 < w_1*h_1:
                flag[idx_2] = False

    for idx, f in enumerate(flag):
        if f:
            new_contours.append(contours[idx])

    return new_contours


def judge_contour_area(w, h, low_threshold, high_threshold):
    area = w * h
    if low_threshold < area < high_threshold:
        return True
    return False


if __name__ == '__main__':
    main()
