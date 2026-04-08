#include <opencv2/opencv.hpp>
#include <iostream>
#include <sys/stat.h>

int main() {
    std::string save_dir = "../data/image/chessboard_10_7_paper_st_1";
    mkdir(save_dir.c_str(), 0777);

    cv::VideoCapture left_cam(2, cv::CAP_V4L2);
    cv::VideoCapture right_cam(4, cv::CAP_V4L2);

    if (!left_cam.isOpened() || !right_cam.isOpened()) {
        std::cerr << "Error: Cannot open cameras" << std::endl;
        return 1;
    }

    left_cam.set(cv::CAP_PROP_BUFFERSIZE, 1);
    right_cam.set(cv::CAP_PROP_BUFFERSIZE, 1);

    cv::Mat frameL, frameR, combined;
    int count = 1;

    while (true) {
        for (int i = 0; i < 5; i++) {
            left_cam.grab();
            right_cam.grab();
        }

        bool retL = left_cam.retrieve(frameL);
        bool retR = right_cam.retrieve(frameR);

        if (!retL || !retR) {
            continue;
        }

        cv::hconcat(frameL, frameR, combined);
        cv::imshow("Press SPACE to capture", combined);

        int key = cv::waitKey(1);
        if (key == 32) {
            cv::imwrite(save_dir + "/left" + std::to_string(count) + ".png", frameL);
            cv::imwrite(save_dir + "/right" + std::to_string(count) + ".png", frameR);
            std::cout << "Saved pair " << count << std::endl;
            count++;
        } else if (key == 'q') {
            break;
        }
    }

    left_cam.release();
    right_cam.release();
    cv::destroyAllWindows();

    return 0;
}