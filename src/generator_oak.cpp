#include <iostream>
#include <string>
#include <sys/stat.h>
#include "depthai/depthai.hpp"
#include <opencv2/opencv.hpp>

int main() {
    std::string save_dir = "../data/image/chessboard_10_7_paper_st_1";
    mkdir(save_dir.c_str(), 0777);

    dai::Pipeline pipeline;

    auto monoLeft = pipeline.create<dai::node::MonoCamera>();
    auto monoRight = pipeline.create<dai::node::MonoCamera>();

    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_720_P);
    monoLeft->setBoardSocket(dai::CameraBoardSocket::CAM_B);
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_720_P);
    monoRight->setBoardSocket(dai::CameraBoardSocket::CAM_C);

    auto xoutLeft = pipeline.create<dai::node::XLinkOut>();
    auto xoutRight = pipeline.create<dai::node::XLinkOut>();
    xoutLeft->setStreamName("left");
    xoutRight->setStreamName("right");

    monoLeft->out.link(xoutLeft->input);
    monoRight->out.link(xoutRight->input);

    dai::Device device(pipeline);

    auto qLeft = device.getOutputQueue("left", 4, false);
    auto qRight = device.getOutputQueue("right", 4, false);

    int count = 1;

    while (true) {
        auto inLeft = qLeft->get<dai::ImgFrame>();
        auto inRight = qRight->get<dai::ImgFrame>();

        cv::Mat frameLeft(inLeft->getHeight(), inLeft->getWidth(), CV_8UC1, inLeft->getData().data());
        cv::Mat frameRight(inRight->getHeight(), inRight->getWidth(), CV_8UC1, inRight->getData().data());

        cv::Mat combined;
        cv::hconcat(frameLeft, frameRight, combined);
        cv::imshow("Press SPACE to capture | 'q' to quit", combined);

        int key = cv::waitKey(1);
        if (key == 32) {
            cv::imwrite(save_dir + "/left" + std::to_string(count) + ".png", frameLeft);
            cv::imwrite(save_dir + "/right" + std::to_string(count) + ".png", frameRight);
            std::cout << "Saved pair " << count << std::endl;
            count++;
        } else if (key == 'q') {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}