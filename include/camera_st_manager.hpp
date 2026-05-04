#pragma once


#include <iostream>
#include <opencv2/opencv.hpp>

struct CameraSTManagerParams{
    int video_id;
    cv::Size size_im;
    int api_preference;
    bool use_mjpg;

    CameraSTManagerParams(const std::string& config_name = "../.config/params.yml"){
        cv::FileStorage fs(config_name, cv::FileStorage::READ);
        cv::FileNode node = fs["camera"];

        node["video_id"] >> video_id;
        node["size_im"] >> size_im;
        node["api_preference"] >> api_preference;
        node["use_mjpg"] >> use_mjpg;
    }
};

class CameraSTManager{
    private:
        int video_id_;
        cv::VideoCapture cap_;
        cv::Size size_im_;
        cv::Mat frame_left_, frame_right_;
    public:
        CameraSTManager(const CameraSTManagerParams& p = CameraSTManagerParams()) 
            : video_id_(p.video_id), size_im_(p.size_im), cap_(p.video_id, p.api_preference) {
            if(p.use_mjpg)
                cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G')); //buffer
            if(!p.size_im.empty()){
                cap_.set(cv::CAP_PROP_FRAME_WIDTH, p.size_im.width*2);
                cap_.set(cv::CAP_PROP_FRAME_HEIGHT, p.size_im.height);
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