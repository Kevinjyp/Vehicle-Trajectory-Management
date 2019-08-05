import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# Read until video is completed

# frameList = []
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     frameList.append(frame)
#
# print(len(frameList))
while(True):
    cap = cv2.VideoCapture(r'G:\留学\MIT暑研\视频\RoadAccidents004_x264.mp4')
    gray_sum_list = []

    if (cap.isOpened() == False):
        print("Error opening video stream")
    frameNum = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frameNum += 1
        print(frameNum)
        if ret:
            tempframe = frame
            if (frameNum == 1):
                prepreviousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
            if (frameNum == 2):
                previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
            if (frameNum > 2):
                currentframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
                dif1 = cv2.absdiff(currentframe, previousframe)
                dif2 = cv2.absdiff(prepreviousframe, previousframe)

                prepreviousframe = previousframe
                previousframe = currentframe

                _, thresh1 = cv2.threshold(dif1, 20, 255, cv2.THRESH_BINARY)
                _, thresh2 = cv2.threshold(dif2, 20, 255, cv2.THRESH_BINARY)

                binary = cv2.bitwise_and(dif1, dif2)  # 与运算

                _, binary = cv2.threshold(binary, 20, 255, cv2.THRESH_BINARY)

                # median = cv2.medianBlur(binary, 3)

                ret, threshold_frame = cv2.threshold(currentframe, 20, 255, cv2.THRESH_BINARY)
                gauss_image = cv2.GaussianBlur(threshold_frame, (3, 3), 0)

                cv2.imshow('原图', frame)
                cv2.imshow('Frame', thresh1)

                hist, bins = np.histogram(thresh1.ravel(), 256, [0, 256])
                gray_sum = 0
                gray = 0
                for gray_num in hist:
                    gray_sum += gray * gray_num
                    gray = gray + 1
                gray_sum_list.append(gray_sum)
                if frameNum != 3:
                    gray_different = abs(gray_sum_list[frameNum - 3] - gray_sum_list[frameNum - 4])
                    print(gray_different)
                    i = 0
            # if frameNum != 1:
            #     plt.hist(currentframe.ravel(), 256, [0, 256])
            #     plt.show()
            #     dark_sum = 0
            #     num0_49 = 0
            #     num50_99 = 0
            #     num100_149 = 0
            #     num150_199 = 0
            #     num200_255 = 0
            #     numlist = []
            #
            #     for row in currentframe:
            #         for col in row:
            #             if col < 40:
            #                dark_sum += 1
            #             elif (col < 50):  #col in range[0, 50]:
            #              num0_49 += 1
            #             elif (50 <= col  and col < 100):  #col in range[50, 100]:
            #              num50_99 += 1
            #             elif (100 <= col  and col < 150):  #col in range[100, 150]:
            #              num100_149 += 1
            #             elif (150 <= col  and col < 200):  #col in range[150, 200]:
            #              num150_199 += 1
            #             elif (200 <= col  and col < 255):  #col in range[200, 250]:
            #              num200_255 += 1
            #     print(dark_sum)
            #     print(num0_49)
            #     print(num50_99)
            #     print(num100_149)
            #     print(num150_199)
            #     print(num200_255)
            #     numlist.append(num0_49)
            #     numlist.append(num50_99)
            #     numlist.append(num100_149)
            #     numlist.append(num150_199)
            #     numlist.append(num200_255)
            #     print(numlist)
            #     nplist = np.array(numlist)
            #     meanNum = nplist.mean()
            #     print(meanNum)
        # if frameNum ==200:
          #   break

        if (cv2.waitKey(4) & 0xff == ord("q")) | frameNum >= 300:
            break
    if 0xff == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
