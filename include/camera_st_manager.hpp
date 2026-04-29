#pragma once


#include <iostream>
#include <opencv2/opencv.hpp>

class CameraSTManager{
    private:
        int video_id_;
        cv::VideoCapture cap_;
        cv::Size size_im_;
        cv::Mat frame_left_, frame_right_;
    public:
        CameraSTManager(int video_id, cv::Size size_im=cv::Size(), int api_preference = cv::CAP_ANY, bool use_mjpg = false) 
            : video_id_(video_id), size_im_(size_im), cap_(video_id_, api_preference) {
            if(use_mjpg)
                cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G')); //buffer
            if(!size_im.empty()){
                cap_.set(cv::CAP_PROP_FRAME_WIDTH, size_im.width*2);
                cap_.set(cv::CAP_PROP_FRAME_HEIGHT, size_im.height);
            }
            if (cap_.isOpened()) {
                int real_w = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
                int real_h = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
                size_im_ = cv::Size(real_w / 2, real_h);
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
                cv::cvtColor(frame,frame,cv::COLOR_RGB2GRAY);
                frame_left_ = frame(cv::Rect(0,0,size_im_.width,size_im_.height));
                frame_right_ = frame(cv::Rect(size_im_.width,0,size_im_.width,size_im_.height));
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
            return size_im_;
        }

        bool IsOpened() const {
            return cap_.isOpened();
        }
};