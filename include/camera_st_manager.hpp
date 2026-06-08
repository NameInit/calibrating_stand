#pragma once


#include <iostream>
#include <opencv2/opencv.hpp>

#include "params_manager.hpp"

struct CameraSTManagerParams{
    int &video_id;
    cv::Size &size_im;
    int &api_preference;
    bool &use_mjpg;

    CameraSTManagerParams() :
        video_id(ParamsManager::getInstance()["camera"]["video_id"]),
        size_im(ParamsManager::getInstance()["camera"]["size_im"]),
        api_preference(ParamsManager::getInstance()["camera"]["api_preference"]),
        use_mjpg(ParamsManager::getInstance()["camera"]["use_mjpg"]) {}
};

class CameraSTManager{
    private:
        cv::VideoCapture cap_;
        cv::Size real_size_im_;
        cv::Mat frame_left_, frame_right_;

        CameraSTManagerParams params_;
    public:
        CameraSTManager() {
            cap_ = cv::VideoCapture(params_.video_id, params_.api_preference);

            if(params_.use_mjpg)
                cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G')); //buffer
            if(!params_.size_im.empty()){
                cap_.set(cv::CAP_PROP_FRAME_WIDTH, params_.size_im.width*2);
                cap_.set(cv::CAP_PROP_FRAME_HEIGHT, params_.size_im.height);
            }
            if (cap_.isOpened()) {
                int real_w = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
                int real_h = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
                real_size_im_ = cv::Size(real_w / 2, real_h);
            }
        }
        ~CameraSTManager(){}

        bool Open(){
            return cap_.isOpened();
        }

        bool Read(){
            if(!cap_.isOpened()){
                return false;
            }

            try{
                cv::Mat frame;
                cap_.read(frame);

                frame_left_ = frame(cv::Rect(0,0,real_size_im_.width,real_size_im_.height));
                frame_right_ = frame(cv::Rect(real_size_im_.width,0,real_size_im_.width,real_size_im_.height));
                return true;

            } catch (const std::exception& e) {
                std::cerr << "Error reading frame: " << e.what() << std::endl;
                return false;
            }
        }

        cv::Mat GetLeftFrame() const {
            return frame_left_.clone();
        }

        cv::Mat GetRightFrame() const {
            return frame_right_.clone();
        }

        cv::Mat GetCombinedFrame() const {
            cv::Mat combined;
            cv::hconcat(frame_left_, frame_right_, combined);
            return combined;
        }

        cv::Size getSizeImage(){
            return real_size_im_;
        }

        bool IsOpened() const {
            return cap_.isOpened();
        }

        CameraSTManagerParams& getParams(){
            return params_;
        }
};