#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <vector>
#include <cmath>

#include "camera_oak_manager.hpp"
#include "camera_st_manager.hpp"
#include "stereo_sgbm.hpp"
#include "fps_counter.hpp"
#include "velocity_tracker.hpp"
#include "roi_manager.hpp"

class StereoPipeline {
private:
    CameraSTManager camera_;
    StereoSGBM stereoSGBM_;
    FPSCounter fps_counter_;
    VelocityTracker velocity_tracker_;
    ROIManager roi_manager_;

    // disparity
    double max_disparity_for_vis_;
    
    // image flow
    cv::Mat prev_left_, prev_right_, cur_left_, cur_right_;
    cv::Mat prev_depth_map, cur_depth_map;

public:
    StereoPipeline(const std::string& calib_file_path,
                   int min_disp, int num_disp, int block_size, int p1, int p2,
                   int uniqueness_ratio, int speckle_ws, int speckle_range,
                   bool use_wls, int wls_lambda, float wls_sigma, int median_blur_size,
                   double max_disparity_for_vis_val)
        : camera_(2,{1280, 800},cv::CAP_V4L2,true),
          stereoSGBM_(calib_file_path),
          fps_counter_(),
          velocity_tracker_(),
          roi_manager_(),
          max_disparity_for_vis_(max_disparity_for_vis_val),
          prev_left_(), prev_right_(), cur_left_(), cur_right_()
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
            cv::Mat display_frame;
            if (processFrame(display_frame)) {
                cv::imshow("Left Rectified", display_frame);
            }
            else{
                std::cerr << "Error process frame" << std::endl;
            }

            int key = cv::waitKey(1);
            if (handleKey(key)) {
                break;
            }
        }

        cv::destroyAllWindows();
        return 0;
    }

private:
    bool processFrame(cv::Mat& output_display_frame) {
        if (!camera_.IsOpened()) return false;
        if (!camera_.Read()) return false;

        fps_counter_.update();

        cv::Mat left_raw = camera_.GetLeftFrame();
        cv::Mat right_raw = camera_.GetRightFrame();

        if (left_raw.empty() || right_raw.empty()) return false;

        if(left_raw.channels() == 3) cv::cvtColor(left_raw, left_raw, cv::COLOR_BGR2GRAY);
        if(right_raw.channels() == 3) cv::cvtColor(right_raw, right_raw, cv::COLOR_BGR2GRAY);

        cur_left_ = left_raw;
        cur_right_ = right_raw;

        stereoSGBM_.Rectify(cur_left_, cur_right_, cur_left_, cur_right_);
        cv::cvtColor(cur_left_, output_display_frame, cv::COLOR_GRAY2BGR);

        if(!roi_manager_.CheckInitConstraint()){
            roi_manager_.SetConstraint({0,0}, {output_display_frame.cols, output_display_frame.rows});
        }

        cv::Rect roi = roi_manager_.GetRect();

        cv::Mat disp_map = stereoSGBM_.Compute(cur_left_, cur_right_);
        cv::Mat disp_roi = disp_map(roi);
        
        cur_depth_map = stereoSGBM_.GetDepthMap(disp_map);

        velocity_tracker_.CalcVelocity(prev_left_, cur_left_, prev_depth_map);
        double cur_vel_kmh = velocity_tracker_.GetSmoothed();

        // Отрисовка ROI и информации
        cv::Mat disp_vis_roi;
        double alpha = 255.0 / max_disparity_for_vis_;
        disp_roi.convertTo(disp_vis_roi, CV_8U, alpha);
        cv::applyColorMap(disp_vis_roi, disp_vis_roi, cv::COLORMAP_JET);
        disp_vis_roi.copyTo(output_display_frame(roi));

        cv::Mat depth_roi = stereoSGBM_.GetDepthMap(disp_roi);
        float dist = depth_roi.at<float>(depth_roi.rows / 2, depth_roi.cols / 2);
        
        // вывод
        std::string label = cv::format("FPS: %.1lf | DIST: %.2f m | VEL: %.1f km/h", 
                                        fps_counter_.get_fps(), dist, cur_vel_kmh);
        cv::putText(output_display_frame, label, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 3);

        prev_left_ = cur_left_.clone();
        prev_right_ = cur_right_.clone();
        prev_depth_map = cur_depth_map.clone();

        cv::rectangle(output_display_frame, roi, cv::Scalar(255, 255, 255), 1);

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