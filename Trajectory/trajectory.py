import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import json
import os

base_path = 'E:/MIT'
video_list = ['car_surveillnace.avi',
              'Loop.North.Zhongshan-East-G-1-20141028075000-20141028075916-1657796.ts',
              'Loop.North.Zhongshan-West-G-1-20141028075000-20141028075941-1716171.ts',
              'video1.mp4', # 3
              'video2.mp4',
              'video3.mp4',
              'video4.mp4',
              'video5.mp4',
              'video6.mp4'
              ]
video_name = video_list[2]

path = os.path.join(base_path, 'Video', video_name)

frame_num = 0
current_frame = 0
previous_frame = 0
preprevious_frame = 0
background_frame = 0
gray_sum = 0
gray_sum_list_left_lane = []
vehicle_info_all = {}


def main():
    global frame_num, previous_frame, preprevious_frame

    # while(True):
    cap = cv2.VideoCapture(path)
    #     gray_sum_list = []

    if not cap.isOpened():
        print('Error opening video stream')

    fgbg = cv2.createBackgroundSubtractorMOG2()

    while cap.isOpened():

        # Capture frame-by-frame
        ret, source_img = cap.read()
        frame_num = frame_num + 1
        print(frame_num)

        if ret:
            if frame_num == 1:
                preprevious_frame = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite('messigray.png', source_img)
            if frame_num == 2:
                previous_frame = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
            if frame_num > 2:
                current_frame = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

                # Get Foreground Image
                # binary_foregourd_img = CDI(preprevious_frame, previous_frame, current_frame)
                binary_foregourd_img = GMM(source_img, fgbg)

                # Frame Update
                preprevious_frame = previous_frame
                previous_frame = current_frame

                # vehicle counting
                # gray_sum = 0
                # for y in range(300, 390):
                #     gray_sum += binary_foregourd_img[340, y]
                # gray_sum_list_left_lane.append(gray_sum)

                # Morphological Operation
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                morphology_img = cv2.morphologyEx(binary_foregourd_img, cv2.MORPH_OPEN, kernel)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
                morphology_img = cv2.morphologyEx(morphology_img, cv2.MORPH_CLOSE, kernel)

                # Decide Which Picture to Use
                processed_img = morphology_img
                # processed_img = binary_foregourd_img

                # Bluring
                processed_img = cv2.medianBlur(processed_img, 5)

                # Draw Contours
                source_img = draw_contours(processed_img, source_img)

                # Get Moment Information of Image
                vehicle_info = get_img_moment(processed_img, frame_num, vehicle_info_all)

                # Graw Central Point and Theta
                for key, value in vehicle_info.items():
                    center = (int(vehicle_info[key]['cx']), int(vehicle_info[key]['cy']))
                    cv2.circle(source_img, center, 2, (0, 0, 255))
                    height = 50 * math.tan(math.radians(vehicle_info[key]['theta']))
                    start = (int(vehicle_info[key]['cx'] - 25), int(vehicle_info[key]['cy'] + height))
                    end = (int(vehicle_info[key]['cx'] + 25), int(vehicle_info[key]['cy'] - height))
                    cv2.line(source_img, start, end, (0, 255, 0), thickness=3)

                # Image Show
                frame_list = [source_img, processed_img]
                name_list = ['source_img', 'processed_img']
                img_show_list(name_list, frame_list)

                # # Save Theta on the Picture
                # processed_path = os.path.join(base_path, 'Picture', video_name, 'Processed')
                # source_path = os.path.join(base_path, 'Picture', video_name, 'Source')
                # my_save_picture(processed_path, processed_img, frame_num)
                # my_save_picture(source_path, source_img, frame_num)

        if (cv2.waitKey(1) & 0xff == ord('q')) | frame_num >= 5000:
            break

    # Save Vehicle Position and Direction Information
    j = json.dumps(vehicle_info_all, indent=1)
    data_path = os.path.join(base_path, 'Processed Data', video_name.split('.')[0] + '.json')
    f = open(data_path, 'w')
    f.write(j)
    f.close()


def my_save_picture(path, img, frame_num):
    if not os.path.exists(path):
        os.mkdir(path)
    cv2.imwrite(os.path.join(path, str(frame_num) + '.jpg'), img)


def GMM(frame, fgbg):
    # Backgroung Learning
    fgmask = fgbg.apply(frame, learningRate=0.1)

    # Threshold
    _, fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)

    return fgmask


def CDI(preprevious_frame, previous_frame, current_frame):
    dif1 = cv2.absdiff(current_frame, previous_frame)
    dif2 = cv2.absdiff(preprevious_frame, previous_frame)

    _, thresh1 = cv2.threshold(dif1, 20, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(dif2, 20, 255, cv2.THRESH_BINARY)

    binary_dif_cdi = cv2.bitwise_and(thresh1, thresh2)  # 与运算

    return binary_dif_cdi


def get_img_moment(img, frame_num, vehicle_info_all):
    # Get Moment Information
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    vehicle_info_frame = {}
    car_num = 0

    for cnt in contours:
        # area, height...
        (x, y, w, h) = cv2.boundingRect(cnt)
        if not judge_contour(x, y, w, h):
            continue

        vehicle_info = {}
        M = cv2.moments(cnt)

        vehicle_info['cx'] = M['m10'] / M['m00']
        vehicle_info['cy'] = M['m01'] / M['m00']

        mu11_prime = M['mu11'] / M['m00']
        mu20_prime = M['mu20'] / M['m00']
        mu02_prime = M['mu02'] / M['m00']

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


def draw_contours(binary_img, source_img):
    # Draw Contours
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # real contours
    # cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)
    # rect contours
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        if judge_contour(x, y, w, h):
            cv2.rectangle(source_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return source_img


def judge_contour(x, y, w, h):
    area = w * h
    if 600 < area < 10000:
        return True
    return False


if __name__ == '__main__':
    main()