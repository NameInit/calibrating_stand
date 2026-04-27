#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <vector>
#include <cmath>

#include "camera_st_manager.hpp"
#include "stereo_sgbm.hpp"
#include "fps_counter.hpp"
#include "velocity_tracker.hpp"
#include "roi_manager.hpp"
#include "vizualizer.hpp"

class StereoPipeline {
private:
    CameraSTManager camera_;
    StereoSGBM stereoSGBM_;
    FPSCounter fps_counter_;
    VelocityTracker velocity_tracker_;
    ROIManager roi_manager_;
    Vizualizer vizualizer_;
    
    // image flow
    cv::Mat prev_left_, prev_right_, cur_left_, cur_right_;
    cv::Mat prev_depth_map_, cur_depth_map_;

    cv::Mat cur_disp_map_;

public:
    StereoPipeline(const std::string& calib_file_path,
                   int min_disp, int num_disp, int block_size, int p1, int p2,
                   int uniqueness_ratio, int speckle_ws, int speckle_range,
                   bool use_wls, int wls_lambda, float wls_sigma, int median_blur_size) 
        : camera_(2,{1280, 800},cv::CAP_V4L2,true),
          stereoSGBM_(calib_file_path),
          fps_counter_(),
          velocity_tracker_(),
          roi_manager_(),
          vizualizer_("Left",num_disp)
    {
        stereoSGBM_.Create(min_disp, num_disp, block_size, p1, p2,
                          uniqueness_ratio, speckle_ws, speckle_range,
                          use_wls, wls_lambda, wls_sigma, median_blur_size);

        velocity_tracker_.Init(stereoSGBM_.GetMatrixLeft());
        velocity_tracker_.Start(); //старт таймера внутри для подсчёта dt
    }

    int run() {
        if (!camera_.IsOpened()) {
            std::cerr << "Failed to open camera" << std::endl;
            return 1;
        }

        while (true) {
            if (processFrame()) {
                vizualizer_.show();
            }
            else {
                std::cerr << "Error process frame" << std::endl;
            }
            
            int key = cv::waitKey(1);
            if (handleKey(key)) {
                break;
            }
        }

        return 0;
    }

private:
    bool processFrame() {
        if (!camera_.IsOpened()) return false;
        if (!camera_.Read()) return false;

        fps_counter_.update();

        cv::Mat left_raw = camera_.GetLeftFrame();
        cv::Mat right_raw = camera_.GetRightFrame();

        if(left_raw.empty() || right_raw.empty()) {
            std::cerr << "Empty frame" << std::endl;
            return false;
        }

        if(left_raw.channels()>1 || right_raw.channels()>1){
            std::cerr << "Error len channels in frame";
            return false;
        }

        cur_left_ = left_raw;
        cur_right_ = right_raw;

        stereoSGBM_.Rectify(cur_left_, cur_right_, cur_left_, cur_right_);

        if(!roi_manager_.CheckInitConstraint()){
            roi_manager_.SetConstraint({0,0}, {cur_left_.cols, cur_left_.rows});
        }

        //fps
        double fps=fps_counter_.get_fps();

        //roi
        cv::Rect roi = roi_manager_.GetRect();

        //disparity
        cur_disp_map_ = stereoSGBM_.Compute(cur_left_, cur_right_);
        cv::Mat disp_roi = cur_disp_map_(roi);
        
        //distance
        cur_depth_map_ = stereoSGBM_.GetDepthMap(cur_disp_map_);
        
        //roi
        cv::Mat depth_roi = cur_depth_map_(roi);
        float dist = depth_roi.at<float>(depth_roi.rows / 2, depth_roi.cols / 2);

        //velocity
        velocity_tracker_.CalcVelocity(prev_left_, cur_left_, prev_depth_map_);
        double cur_vel_kmh = velocity_tracker_.GetSmoothed();

        vizualizer_.render(cur_left_, cur_disp_map_, roi, fps, dist, cur_vel_kmh);
        

        prev_left_ = cur_left_.clone();
        prev_right_ = cur_right_.clone();
        prev_depth_map_ = cur_depth_map_.clone();
        return true;
    }

    bool handleKey(int key) {
        if (key == 'a' || key == 'A') roi_manager_.moveROI(-30, 0);
        else if (key == 'd' || key == 'D') roi_manager_.moveROI(30, 0);
        else if (key == 'w' || key == 'W') roi_manager_.moveROI(0, -30);
        else if (key == 's' || key == 'S') roi_manager_.moveROI(0, 30);
        else if (key == 'c' || key == 'C') roi_manager_.resizeROI(-30, 0);
        else if (key == 'v' || key == 'V') roi_manager_.resizeROI(30, 0);
        else if (key == 'z' || key == 'Z') roi_manager_.resizeROI(0, -30);
        else if (key == 'x' || key == 'X') roi_manager_.resizeROI(0, 30);

        return key == 'q' || key == 27; // 'q' or 'ESC'
    }
};