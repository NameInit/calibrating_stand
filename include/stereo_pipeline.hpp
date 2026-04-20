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
#include "stereo_sgbm.hpp"
#include "fps_counter.hpp"
#include "velocity_tracker.hpp"

class StereoPipeline {
private:
    CameraOAKManager camera_;
    StereoSGBM stereoSGBM_;
    FPSCounter fps_counter_;
    VelocityTracker velocity_tracker_;

    // roi
    int cur_x_;
    int cur_y_;
    int win_width_;
    int win_height_;

    // disparity
    double max_disparity_for_vis_;
    
    // image flow
    cv::Mat prev_left_, prev_right_, cur_left_, cur_right_;
    cv::Mat prev_depth_map, cur_depth_map;

    // time for calc dt, velocity
    std::chrono::time_point<std::chrono::steady_clock> prev_time_;
    double smoothed_velocity_;

public:
    StereoPipeline(const std::string& calib_file_path,
                   int min_disp, int num_disp, int block_size, int p1, int p2,
                   int uniqueness_ratio, int speckle_ws, int speckle_range,
                   bool use_wls, int wls_lambda, float wls_sigma, int median_blur_size,
                   int initial_x, int initial_y, int initial_width, int initial_height,
                   double max_disparity_for_vis_val)
        : camera_(),
          stereoSGBM_(calib_file_path),
          fps_counter_(),
          velocity_tracker_(),
          cur_x_(initial_x), cur_y_(initial_y),
          win_width_(initial_width), win_height_(initial_height),
          max_disparity_for_vis_(max_disparity_for_vis_val),
          prev_left_(), prev_right_(), cur_left_(), cur_right_(),
          smoothed_velocity_(0.0)
    {
        stereoSGBM_.Create(min_disp, num_disp, block_size, p1, p2,
                          uniqueness_ratio, speckle_ws, speckle_range,
                          use_wls, wls_lambda, wls_sigma, median_blur_size);

        velocity_tracker_.Init(stereoSGBM_.GetMatrixLeft());
        velocity_tracker_.Start(); //старт таймера внутри для подсчёта dt
        prev_time_ = std::chrono::steady_clock::now();
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
    float getSubpixelDepth(const cv::Mat& depth, cv::Point2f pt) {
        int x = std::floor(pt.x);
        int y = std::floor(pt.y);
        
        if (x < 0 || y < 0 || x >= depth.cols - 1 || y >= depth.rows - 1) return 0.0f;

        float dx = pt.x - x;
        float dy = pt.y - y;

        float d00 = depth.at<float>(y, x);
        float d10 = depth.at<float>(y, x + 1);
        float d01 = depth.at<float>(y + 1, x);
        float d11 = depth.at<float>(y + 1, x + 1);

        if (d00 <= 0 || d10 <= 0 || d01 <= 0 || d11 <= 0) return 0.0f;

        return d00 * (1 - dx) * (1 - dy) + d10 * dx * (1 - dy) + 
               d01 * (1 - dx) * dy + d11 * dx * dy;
    }

    bool processFrame(cv::Mat& output_display_frame) {
        if (!camera_.IsOpened()) return false;
        if (!camera_.Read()) return false;

        fps_counter_.update();

        cv::Mat left_raw = camera_.GetLeftFrame();
        cv::Mat right_raw = camera_.GetRightFrame();

        if (left_raw.empty() || right_raw.empty()) return false;

        if(left_raw.channels() == 3) cv::cvtColor(left_raw, left_raw, cv::COLOR_BGR2GRAY);
        if(right_raw.channels() == 3) cv::cvtColor(right_raw, right_raw, cv::COLOR_BGR2GRAY);

        stereoSGBM_.Rectify(left_raw, right_raw, cur_left_, cur_right_);
        cv::cvtColor(cur_left_, output_display_frame, cv::COLOR_GRAY2BGR);

        clampROI(output_display_frame.cols, output_display_frame.rows);
        cv::Rect roi(cur_x_ - win_width_/2, cur_y_ - win_height_/2, win_width_, win_height_);

        cv::Mat disp_map = stereoSGBM_.Compute(cur_left_, cur_right_);
        cv::Mat disp_roi = disp_map(roi);
        
        cur_depth_map = stereoSGBM_.GetDepthMap(disp_map);

        velocity_tracker_.CalcVelocity(prev_left_, cur_left_, prev_depth_map);
        double cur_vel_kmh = velocity_tracker_.GetSmoothed();

        // Отрисовка ROI и информации
        if (!disp_roi.empty()) {
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
        }

        prev_left_ = cur_left_.clone();
        prev_right_ = cur_right_.clone();
        prev_depth_map = cur_depth_map.clone();

        cv::rectangle(output_display_frame, roi, cv::Scalar(255, 255, 255), 1);

        return true;
    }

    bool handleKey(int key) {
        if (key == 'a' || key == 'A') moveROI(-30, 0);
        else if (key == 'd' || key == 'D') moveROI(30, 0);
        else if (key == 'w' || key == 'W') moveROI(0, -30);
        else if (key == 's' || key == 'S') moveROI(0, 30);
        else if (key == 'c' || key == 'C') resizeROI(-30, 0);
        else if (key == 'v' || key == 'V') resizeROI(30, 0);
        else if (key == 'z' || key == 'Z') resizeROI(0, -30);
        else if (key == 'x' || key == 'X') resizeROI(0, 30);

        return key == 'q' || key == 27; // 'q' or 'ESC'
    }

    void clampROI(int img_cols, int img_rows) {
        win_width_ = std::max(100, std::min(win_width_, img_cols));
        win_height_ = std::max(100, std::min(win_height_, img_rows));

        cur_x_ = std::max(win_width_ / 2, std::min(cur_x_, img_cols - win_width_ / 2));
        cur_y_ = std::max(win_height_ / 2, std::min(cur_y_, img_rows - win_height_ / 2));
    }

    double CalcVelocity(const cv::Mat& prev_img, const cv::Mat& cur_img, const cv::Mat& prev_depth, double dt){
        if (prev_img.empty() || cur_img.empty() || prev_depth.empty() || dt <= 0.0) 
            return 0.0;

        std::vector<cv::Point2f> pts_prev, pts_curr, pts_prev_back;
        
        // маска на будущее, если нужно выкинуть часть потока изображений
        cv::Mat mask; 
        cv::goodFeaturesToTrack(prev_img, pts_prev, 400, 0.01, 10, mask);
        if (pts_prev.empty()) return 0.0;

        std::vector<uchar> status_fwd, status_bwd;
        std::vector<float> err_fwd, err_bwd;

        //прямой и обратный матчер для текущего и прошлого кадра
        cv::calcOpticalFlowPyrLK(prev_img, cur_img, pts_prev, pts_curr, status_fwd, err_fwd);
        cv::calcOpticalFlowPyrLK(cur_img, prev_img, pts_curr, pts_prev_back, status_bwd, err_bwd);

        cv::Mat K = stereoSGBM_.GetMatrixLeft(); 
        float fx = K.at<double>(0, 0);
        float fy = K.at<double>(1, 1);
        float cx = K.at<double>(0, 2);
        float cy = K.at<double>(1, 2);

        std::vector<cv::Point3f> object_points;
        std::vector<cv::Point2f> image_points;

        for (size_t i = 0; i < pts_prev.size(); i++) {
            if (status_fwd[i] && status_bwd[i]) {
                float dx = pts_prev[i].x - pts_prev_back[i].x;
                float dy = pts_prev[i].y - pts_prev_back[i].y;
                
                // ошибка возврата = 1 пиксель
                if (dx * dx + dy * dy < 1.0f) {
                    
                    float z = getSubpixelDepth(prev_depth, pts_prev[i]);
                    
                    if (z > 0.5f && z < 20.0f) {
                        float x = (pts_prev[i].x - cx) * z / fx;
                        float y = (pts_prev[i].y - cy) * z / fy;
                        
                        object_points.push_back(cv::Point3f(x, y, z));
                        image_points.push_back(pts_curr[i]);
                    }
                }
            }
        }

        if (object_points.size() < 12) return 0.0;

        cv::Mat rvec, tvec;
        cv::Mat ind_true_points;
        bool success = cv::solvePnPRansac(object_points, image_points, K, cv::Mat(), 
                                          rvec, tvec, false, 200, 3.0f, 0.99, ind_true_points);

        float inlier_ratio = (float)ind_true_points.rows / (float)object_points.size();
        if (!success || inlier_ratio < 0.2f || ind_true_points.rows < 15) {
            return 0.0;
        }

        // уточнение позиции
        std::vector<cv::Point3f> inlier_obj;
        std::vector<cv::Point2f> inlier_img;
        for (int i = 0; i < ind_true_points.rows; i++) {
            int idx = ind_true_points.at<int>(i);
            inlier_obj.push_back(object_points[idx]);
            inlier_img.push_back(image_points[idx]);
        }
        
        cv::solvePnPRefineLM(inlier_obj, inlier_img, K, cv::Mat(), rvec, tvec);

        double dist_travelled = cv::norm(tvec);
        return dist_travelled / dt; // м/с
    }
};