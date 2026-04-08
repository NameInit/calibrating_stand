import cv2
import os
import time

left_cam = cv2.VideoCapture(2, cv2.CAP_V4L2)
right_cam = cv2.VideoCapture(4, cv2.CAP_V4L2)

left_cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
right_cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

count = 1
while True:
    for _ in range(5):
        left_cam.grab()
        right_cam.grab()
    
    retL, frameL = left_cam.retrieve()
    retR, frameR = right_cam.retrieve()
    
    if not retL or not retR:
        continue
    
    cv2.imshow("Press SPACE to capture", cv2.hconcat([frameL, frameR]))
    
    key = cv2.waitKey(1)
    if key == 32:  # Пробел
        timestamp = time.time()
        cv2.imwrite(f"../data/image/chessboard_10_7_paper_st_1/left{count}.png", frameL)
        cv2.imwrite(f"../data/image/chessboard_10_7_paper_st_1/right{count}.png", frameR)
        print(f"Saved pair {count}")
        count += 1
    elif key == ord('q'):
        break

left_cam.release()
right_cam.release()
cv2.destroyAllWindows()