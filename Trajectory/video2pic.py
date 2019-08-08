import cv2
import os


def save_img():
    base_path = 'E:/MIT'
    video_name = 'car_surveillnace.avi'
    # video_name = 'Loop.North.Zhongshan-East-G-1-20141028075000-20141028075916-1657796.ts'
    video_path = os.path.join(base_path, 'video', video_name)
    pic_path = os.path.join(base_path, 'Picture', video_name)
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    while cap.isOpened():
        frame_num = frame_num + 1
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(pic_path, str(frame_num) + '.jpg'), frame)
            cv2.waitKey(1)
        else:
            break
    cap.release()
    print('save_success')


def main():
    save_img()


if __name__ == '__main__':
    main()
