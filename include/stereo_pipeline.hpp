#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <vector>
#include <cmath>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "camera_st_manager.hpp"
#include "stereo_sgbm.hpp"
#include "fps_counter.hpp"
#include "velocity_tracker.hpp"
#include "roi_manager.hpp"
#include "vizualizer.hpp"
#include "stereo_optimized.hpp"
#include "stereo_sgbm_optimazed.hpp"


class StereoPipeline {
private:
    CameraSTManager camera_;
    StereoSGBMOptimized stereoSGBM_;
    FPSCounter fps_counter_;
    VelocityTracker velocity_tracker_;
    ROIManager roi_manager_;
    Vizualizer vizualizer_;
    
    //path
    std::string calib_file_path_;

    //image flow
    cv::Mat prev_left_, prev_right_, cur_left_, cur_right_, visual_;
    cv::Mat prev_depth_map_, cur_depth_map_;

    cv::Mat cur_disp_map_;

    //metrics
    double vel_kmh_;

    //thread sgbm
    std::thread sgbm_thread_;
    std::atomic<bool> sgbm_is_runnig_;
    std::mutex sgbm_mutex_in_;
    std::mutex sgbm_mutex_out_;

    //thread velocity
    std::thread vel_thread_;
    std::atomic<bool> vel_is_running_;
    std::mutex vel_mutex_in_;

    //shared fields
    cv::Mat sgbm_shared_left_, sgbm_shared_right_;
    cv::Mat sgbm_shared_disp_map_, sgbm_shared_depth_map_;
    cv::Mat vel_shared_cur_left_, vel_shared_prev_left_, vel_shared_prev_depth_map_; 

public:
    StereoPipeline(const std::string& calib_file_path) 
        : camera_(),
          stereoSGBM_(calib_file_path),
          fps_counter_(),
          velocity_tracker_(),
          roi_manager_(),
          vizualizer_(),
          calib_file_path_(calib_file_path)
    {
        stereoSGBM_.create();

        velocity_tracker_.initMatrixCam(stereoSGBM_.getMatrixLeft());
        velocity_tracker_.start(); //старт таймера внутри для подсчёта dt

        roi_manager_.setConstraint({1,1}, camera_.getSizeImage());
    }

    ~StereoPipeline(){
        stopSGBMLoop_();
        stopVelocityLoop_();
    }

    StereoPipeline& refresh(const std::string& calib_file_path){
        stopSGBMLoop_();
        stopVelocityLoop_();

        ParamsManager::getInstance().refresh();
        stereoSGBM_.load(calib_file_path);
        stereoSGBM_.create();
        
        velocity_tracker_.initMatrixCam(stereoSGBM_.getMatrixLeft());
        velocity_tracker_.start();

        {
            std::lock_guard<std::mutex> lock1(sgbm_mutex_in_);
            std::lock_guard<std::mutex> lock2(sgbm_mutex_out_);
            sgbm_shared_left_.release();
            sgbm_shared_right_.release();
            sgbm_shared_disp_map_.release();
            sgbm_shared_depth_map_.release();
        }
        {
            std::lock_guard<std::mutex> lock3(vel_mutex_in_);
            vel_shared_prev_left_.release();
            vel_shared_cur_left_.release();
            vel_shared_prev_depth_map_.release();
        }

        sgbm_is_runnig_ = true;
        sgbm_thread_ = std::thread(&StereoPipeline::runSGBMLoop_, this);

        vel_is_running_ = true;
        vel_thread_ = std::thread(&StereoPipeline::runVelocityLoop_, this);
        return *this;
    }

    int run() {
        if (!camera_.IsOpened()) {
            std::cerr << "Failed to open camera" << std::endl;
            return 1;
        }

        sgbm_is_runnig_=true;
        sgbm_thread_ = std::thread(&StereoPipeline::runSGBMLoop_, this);

        vel_is_running_=true;
        vel_thread_ = std::thread(&StereoPipeline::runVelocityLoop_, this);

        while (true) {
            if (processFrame_()) {
                vizualizer_.show();
            }
            else {
                std::cerr << "Error process frame" << std::endl;
            }
            
            int key = cv::waitKey(1);
            if (handleKey_(key)) {
                break;
            }
        }

        return 0;
    }

private:
    bool processFrame_() {
        if (!camera_.IsOpened()) return false;
        if (!camera_.Read()) return false;

        fps_counter_.update();
        double fps=fps_counter_.get_fps();

        visual_ = camera_.GetLeftFrame();
        cur_left_ = camera_.GetLeftFrame();
        cur_right_ = camera_.GetRightFrame();

        checkerChannenls(cur_left_, cur_right_);

        stereoSGBM_.rectify(cur_left_, cur_right_, cur_left_, cur_right_);

        //send frames for calc stereo
        {
            std::lock_guard<std::mutex> lock(sgbm_mutex_in_);
            sgbm_shared_left_=cur_left_.clone();
            sgbm_shared_right_=cur_right_.clone();
        }

        //get stereo maps
        {
            std::lock_guard<std::mutex> lock(sgbm_mutex_out_);
            if(sgbm_shared_depth_map_.empty()||sgbm_shared_disp_map_.empty()){
                return true;
            }
            cur_disp_map_=sgbm_shared_disp_map_.clone();
            cur_depth_map_=sgbm_shared_depth_map_.clone();
        }

        //roi and distance to center roi
        cv::Rect roi = roi_manager_.getRect();
        cv::Mat depth_roi = cur_depth_map_(roi);
        float dist = depth_roi.at<float>(depth_roi.rows / 2, depth_roi.cols / 2);

        //send frames for calc velocity
        {
            std::lock_guard<std::mutex> lock(vel_mutex_in_);
            vel_shared_prev_left_ = prev_left_.clone();
            vel_shared_cur_left_ = cur_left_.clone();
            vel_shared_prev_depth_map_ = prev_depth_map_.clone();
        }

        vizualizer_.render(visual_, cur_disp_map_, roi, fps, dist, vel_kmh_);

        prev_left_ = cur_left_.clone();
        prev_right_ = cur_right_.clone();
        prev_depth_map_ = cur_depth_map_.clone();
        return true;
    }

    bool checkerChannenls(cv::Mat& left, cv::Mat& right){
        if(cur_left_.channels()==3){
            cv::cvtColor(cur_left_,cur_left_,cv::COLOR_RGB2GRAY);
        }

        if(cur_right_.channels()==3){
            cv::cvtColor(cur_right_,cur_right_,cv::COLOR_RGB2GRAY);
        }

        if(cur_left_.empty() || cur_right_.empty()) {
            std::cerr << "Empty frame" << std::endl;
            return false;
        }

        if(cur_left_.channels()>1 || cur_right_.channels()>1){
            std::cerr << "Error len channels in frame";
            return false;
        }
        return true;
    }

    bool handleKey_(int key) {
        if (key == 'a' || key == 'A') roi_manager_.moveROI(-30, 0);
        else if (key == 'd' || key == 'D') roi_manager_.moveROI(30, 0);
        else if (key == 'w' || key == 'W') roi_manager_.moveROI(0, -30);
        else if (key == 's' || key == 'S') roi_manager_.moveROI(0, 30);
        else if (key == 'c' || key == 'C') roi_manager_.resizeROI(-30, 0);
        else if (key == 'v' || key == 'V') roi_manager_.resizeROI(30, 0);
        else if (key == 'z' || key == 'Z') roi_manager_.resizeROI(0, -30);
        else if (key == 'x' || key == 'X') roi_manager_.resizeROI(0, 30);
        else if (key == 'r' || key == 'R') { refresh(calib_file_path_); }

        return key == 'q' || key == 27; // 'q' or 'ESC'
    }

    void runSGBMLoop_(){
        while(sgbm_is_runnig_){
            bool has_frames = true;
            cv::Mat local_left, local_right;
            {
                std::lock_guard<std::mutex> lock(sgbm_mutex_in_);
                if(sgbm_shared_left_.empty()||sgbm_shared_right_.empty()){
                    has_frames=false;
                }
                else{
                    local_left=sgbm_shared_left_.clone();
                    local_right=sgbm_shared_right_.clone();
                }
            }

            if(!has_frames){
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            cv::Mat local_disp_map, local_depth_map;
            local_disp_map = stereoSGBM_.compute(local_left, local_right);
            local_depth_map = stereoSGBM_.getDepthMap(local_disp_map);

            {
                std::lock_guard<std::mutex> lock(sgbm_mutex_out_);
                sgbm_shared_disp_map_ = local_disp_map.clone();
                sgbm_shared_depth_map_ = local_depth_map.clone();
            }
        }
        return ;
    }

    void stopSGBMLoop_(){
        sgbm_is_runnig_=false;
        if(sgbm_thread_.joinable()){
            sgbm_thread_.join();
        }
        return ;
    }

    void runVelocityLoop_(){
        while(vel_is_running_){
            bool has_frames = true;
            cv::Mat local_prev_left, local_cur_left, local_prev_depth_map;
            {
                std::lock_guard<std::mutex> lock(vel_mutex_in_);
                if(vel_shared_prev_left_.empty()||vel_shared_cur_left_.empty()||vel_shared_prev_depth_map_.empty()){
                    has_frames=false;
                }
                else{
                    local_prev_left = vel_shared_prev_left_;
                    local_cur_left = vel_shared_cur_left_;
                    local_prev_depth_map = vel_shared_prev_depth_map_;
                }
            }

            if(!has_frames){
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            velocity_tracker_.calcVelocity(local_prev_left, local_cur_left, local_prev_depth_map);
            vel_kmh_ = velocity_tracker_.getSmoothed();
        }
        return ;
    }

    void stopVelocityLoop_(){
        vel_is_running_=false;
        if(vel_thread_.joinable()){
            vel_thread_.join();
        }
        return ;
    }
};