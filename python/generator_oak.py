import depthai as dai
import cv2
import os

save_dir = "../data/image/chessboard_10_7_paper_st_1"
os.makedirs(save_dir, exist_ok=True)

pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)
xoutLeft.setStreamName("left")
xoutRight.setStreamName("right")

monoLeft.out.link(xoutLeft.input)
monoRight.out.link(xoutRight.input)

count = 1

with dai.Device(pipeline) as device:
    qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)

    while True:
        inLeft = qLeft.get()
        inRight = qRight.get()

        frameLeft = inLeft.getCvFrame()
        frameRight = inRight.getCvFrame()

        combined = cv2.hconcat([frameLeft, frameRight])
        cv2.imshow("Press SPACE to capture | 'q' to quit", cv2.resize(combined, (800,400)))

        key = cv2.waitKey(1)
        if key == 32: # space
            cv2.imwrite(f"{save_dir}/left{count}.png", frameLeft)
            cv2.imwrite(f"{save_dir}/right{count}.png", frameRight)
            print(f"Saved pair {count}")
            count += 1
        elif key == ord('q'):
            break

cv2.destroyAllWindows()