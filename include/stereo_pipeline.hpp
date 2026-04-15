#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <vector>

#include "camera_oak_manager.hpp"
#include "stereo_sgbm.hpp"
#include "fps_counter.hpp"
#include "velocity_tracker.hpp"

class StereoPipeline {
private:
    CameraOAKManager camera_;
    StereoSGBM stereoSGBM_;
    FPSCounter fps_counter_;
    VelocityTracker velocity_tracker_;

    //roi
    int cur_x_;
    int cur_y_;
    int win_width_;
    int win_height_;

    //disparity
    double max_disparity_for_vis_;
    
    //image flow
    cv::Mat prev_left_, prev_right_, cur_left_, cur_right_;
public:
    StereoPipeline(const std::string& calib_file_path,
                   int min_disp, int num_disp, int block_size, int p1, int p2,
                   int uniqueness_ratio, int speckle_ws, int speckle_range,
                   bool use_wls, int wls_lambda, float wls_sigma, int median_blur_size,
                   int initial_x, int initial_y, int initial_width, int initial_height,
                   double max_disparity_for_vis_val)
        : camera_(),
          stereoSGBM_(calib_file_path),
          fps_counter_(), velocity_tracker_(),
          cur_x_(initial_x), cur_y_(initial_y),
          win_width_(initial_width), win_height_(initial_height),
          max_disparity_for_vis_(max_disparity_for_vis_val),
          prev_left_(), prev_right_(), cur_left_(), cur_right_()
    {
        stereoSGBM_.Create(min_disp, num_disp, block_size, p1, p2,
                          uniqueness_ratio, speckle_ws, speckle_range,
                          use_wls, wls_lambda, wls_sigma, median_blur_size);
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

            int key = cv::waitKey(1);
            if (handleKey(key)) {
                break;
            }
        }

        cv::destroyAllWindows();
        return 0;
    }

    void moveROI(int dx, int dy) {
        cur_x_ += dx;
        cur_y_ += dy;
    }

    void resizeROI(int d_width, int d_height) {
        win_width_ += d_width;
        win_height_ += d_height;
    }

private:
    bool processFrame(cv::Mat& output_display_frame) {
        if (!camera_.IsOpened()) {
            std::cerr << "Camera not opened during frame processing." << std::endl;
            return false;
        }
        if (!camera_.Read()) {
            std::cerr << "Failed to read frame from camera." << std::endl;
            return false;
        }

        fps_counter_.update();

        cv::Mat left_raw = camera_.GetLeftFrame();
        cv::Mat right_raw = camera_.GetRightFrame();

        if (left_raw.empty() || right_raw.empty()) {
            std::cerr << "Received empty frames." << std::endl;
            return false;
        }

        if(left_raw.channels() == 3){
            cv::cvtColor(left_raw, left_raw, cv::COLOR_BGR2GRAY);
        }
        if(right_raw.channels() == 3){
            cv::cvtColor(right_raw, right_raw, cv::COLOR_BGR2GRAY);
        }

        stereoSGBM_.Rectify(left_raw, right_raw, cur_left_, cur_right_);

        cv::cvtColor(cur_left_, output_display_frame, cv::COLOR_GRAY2BGR);

        clampROI(output_display_frame.cols, output_display_frame.rows);
        cv::Rect roi(cur_x_ - win_width_/2, cur_y_ - win_height_/2, win_width_, win_height_);

        cv::Mat left_roi = cur_left_(roi);
        cv::Mat right_roi = cur_right_(roi);

        cv::Mat disp_roi = stereoSGBM_.Compute(cur_left_, cur_right_);
        disp_roi=disp_roi(roi);

        //считаем карту глубины для roi
        if (!disp_roi.empty()) {
            cv::Mat disp_vis_roi;

            double alpha = 255.0 / max_disparity_for_vis_;
            disp_roi.convertTo(disp_vis_roi, CV_8U, alpha);

            cv::applyColorMap(disp_vis_roi, disp_vis_roi, cv::COLORMAP_JET);
            disp_vis_roi.copyTo(output_display_frame(roi));

            cv::Mat depth_roi = stereoSGBM_.GetDepthMap(disp_roi);
            
            float dist = depth_roi.at<float>(depth_roi.rows / 2, depth_roi.cols / 2);
            double fps = fps_counter_.get_fps();

            std::string label = cv::format("FPS: %.1lf | DIST: %.2f m", fps, dist);
            cv::putText(output_display_frame, label, cv::Point(roi.x, roi.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        }


        //считаем скорость
        if(!prev_left_.empty() && !prev_right_.empty()){
            ;
        }
        prev_left_ = cur_left_.clone();
        prev_right_ = cur_right_.clone();

        cv::rectangle(output_display_frame, roi, cv::Scalar(255, 255, 255), 1);

        return true;
    }

    bool handleKey(int key) {
        bool roi_changed = false;
        if (key == 'a' || key == 'A') { moveROI(-30, 0); roi_changed = true; }
        else if (key == 'd' || key == 'D') { moveROI(30, 0); roi_changed = true; }
        else if (key == 'w' || key == 'W') { moveROI(0, -30); roi_changed = true; }
        else if (key == 's' || key == 'S') { moveROI(0, 30); roi_changed = true; }
        else if (key == 'c' || key == 'C') { resizeROI(-30, 0); roi_changed = true; }
        else if (key == 'v' || key == 'V') { resizeROI(30, 0); roi_changed = true; }
        else if (key == 'z' || key == 'Z') { resizeROI(0, -30); roi_changed = true; }
        else if (key == 'x' || key == 'X') { resizeROI(0, 30); roi_changed = true; }

        return key == 'q';
    }

    void clampROI(int img_cols, int img_rows) {
        win_width_ = std::max(100, std::min(win_width_, img_cols));
        win_height_ = std::max(100, std::min(win_height_, img_rows));

        cur_x_ = std::max(win_width_ / 2, std::min(cur_x_, img_cols - win_width_ / 2));
        cur_y_ = std::max(win_height_ / 2, std::min(cur_y_, img_rows - win_height_ / 2));
    }
};