#include "opencv2/opencv.hpp"
#include <iostream>


int main(){
    cv::VideoCapture cap(2, cv::CAP_V4L2);


    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open cameras" << std::endl;
        return 1;
    }

    cv::Mat frame, left_img, right_img;
    while (cap.read(frame)) {
        int width = frame.cols / 2;
        left_img = frame(cv::Rect(0, 0, width, frame.rows));
        right_img = frame(cv::Rect(width, 0, width, frame.rows));

        cv::imshow("Left", left_img);
        cv::imshow("Right", right_img);
        cv::imshow("ConcatFrame", frame);

        int key = cv::waitKey(1);
        if (key == 'q') break;
    }

    return 0;
}