#pragma once

#include <iostream>
#include <string>
#include <sys/stat.h>
#include "depthai/depthai.hpp"
#include <opencv2/opencv.hpp>

class CameraOAKManager {
private:    
    dai::Pipeline pipeline;
    dai::Device* device;
    std::shared_ptr<dai::DataOutputQueue> q_left;
    std::shared_ptr<dai::DataOutputQueue> q_right;
    cv::Mat frame_left;
    cv::Mat frame_right;
    bool device_ready;

public:
    CameraOAKManager() : device(nullptr), device_ready(false) {
        SetupPipeline();
        Open();
    }

    ~CameraOAKManager() {
        if (device) {
            delete device;
        }
    }

    void SetupPipeline() {
        auto mono_left = pipeline.create<dai::node::MonoCamera>();
        auto mono_right = pipeline.create<dai::node::MonoCamera>();

        mono_left->setResolution(dai::MonoCameraProperties::SensorResolution::THE_720_P);
        mono_left->setBoardSocket(dai::CameraBoardSocket::CAM_B);
        mono_right->setResolution(dai::MonoCameraProperties::SensorResolution::THE_720_P);
        mono_right->setBoardSocket(dai::CameraBoardSocket::CAM_C);

        auto xout_left = pipeline.create<dai::node::XLinkOut>();
        auto xout_right = pipeline.create<dai::node::XLinkOut>();
        xout_left->setStreamName("left");
        xout_right->setStreamName("right");

        mono_left->out.link(xout_left->input);
        mono_right->out.link(xout_right->input);
    }

    bool Open() {
        try {
            device = new dai::Device(pipeline);
            q_left = device->getOutputQueue("left", 4, false);
            q_right = device->getOutputQueue("right", 4, false);
            device_ready = true;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error opening OAK camera: " << e.what() << std::endl;
            device_ready = false;
            return false;
        }
    }

    bool Read() {
        if (!device_ready) {
            return false;
        }

        try {
            auto in_left = q_left->get<dai::ImgFrame>();
            auto in_right = q_right->get<dai::ImgFrame>();

            frame_left = cv::Mat(in_left->getHeight(), in_left->getWidth(), CV_8UC1, in_left->getData().data()).clone();
            frame_right = cv::Mat(in_right->getHeight(), in_right->getWidth(), CV_8UC1, in_right->getData().data()).clone();

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error reading frame: " << e.what() << std::endl;
            return false;
        }
    }

    cv::Mat GetLeftFrame() const {
        return frame_left.clone();
    }

    cv::Mat GetRightFrame() const {
        return frame_right.clone();
    }

    cv::Mat GetCombinedFrame() const {
        cv::Mat combined;
        cv::hconcat(frame_left, frame_right, combined);
        return combined;
    }

    bool IsOpened() const {
        return device_ready;
    }
};