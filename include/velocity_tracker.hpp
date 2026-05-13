#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include "params_manager.hpp"

struct VelocityTrackerParams {
    int &max_corners;
    double &quality_level;
    double &min_distance;
    int &min_pnp_points;
    int &min_inliers;
    double &inlier_ratio_threshold;
    double &ransac_reproj_error;
    int &ransac_iter;
    double &ransac_conf;
    double &min_depth;
    double &max_depth;
    double &smoothed_alpha;

    VelocityTrackerParams() :
        max_corners(ParamsManager::getInstance()["velocity_tracker"]["max_corners"]),
        quality_level(ParamsManager::getInstance()["velocity_tracker"]["quality_level"]),
        min_distance(ParamsManager::getInstance()["velocity_tracker"]["min_distance"]),
        min_pnp_points(ParamsManager::getInstance()["velocity_tracker"]["min_pnp_points"]),
        min_inliers(ParamsManager::getInstance()["velocity_tracker"]["min_inliers"]),
        inlier_ratio_threshold(ParamsManager::getInstance()["velocity_tracker"]["inlier_ratio_threshold"]),
        ransac_reproj_error(ParamsManager::getInstance()["velocity_tracker"]["ransac_reproj_error"]),
        ransac_iter(ParamsManager::getInstance()["velocity_tracker"]["ransac_iter"]),
        ransac_conf(ParamsManager::getInstance()["velocity_tracker"]["ransac_conf"]),
        min_depth(ParamsManager::getInstance()["velocity_tracker"]["min_depth"]),
        max_depth(ParamsManager::getInstance()["velocity_tracker"]["max_depth"]),
        smoothed_alpha(ParamsManager::getInstance()["velocity_tracker"]["smoothed_alpha"]) {}
};

class VelocityTracker {
private:
    cv::Mat K_;
    VelocityTrackerParams params_;
    std::chrono::time_point<std::chrono::steady_clock> prev_time_, cur_time_;
    float fx_, fy_, cx_, cy_;
    double smoothed_velocity_, vel_ms_;

    float getSubpixelDepth_(const cv::Mat& depth, cv::Point2f pt) const {
        int x = static_cast<int>(pt.x);
        int y = static_cast<int>(pt.y);
        if (x < 0 || y < 0 || x >= depth.cols - 1 || y >= depth.rows - 1) return 0.0f;

        float dx = pt.x - x;
        float dy = pt.y - y;

        const float* row0 = depth.ptr<float>(y);
        const float* row1 = depth.ptr<float>(y + 1);

        float d00 = row0[x], d10 = row0[x + 1];
        float d01 = row1[x], d11 = row1[x + 1];

        if (d00 <= 0 || d10 <= 0 || d01 <= 0 || d11 <= 0) return 0.0f;

        return d00 * (1 - dx) * (1 - dy) + d10 * dx * (1 - dy) + 
               d01 * (1 - dx) * dy + d11 * dx * dy;
    }

public:
    VelocityTracker()
     : fx_(0), fy_(0), cx_(0), cy_(0), smoothed_velocity_(0.), vel_ms_(0.) {}

    void InitMatrixCam(const cv::Mat& K) {
        K_ = K.clone();
        fx_ = static_cast<float>(K_.at<double>(0, 0));
        fy_ = static_cast<float>(K_.at<double>(1, 1));
        cx_ = static_cast<float>(K_.at<double>(0, 2));
        cy_ = static_cast<float>(K_.at<double>(1, 2));
    }

    double CalcVelocity(const cv::Mat& prev_img, const cv::Mat& cur_img, 
                        const cv::Mat& prev_depth,
                        const cv::Mat& mask = cv::Mat(),
                        bool to_kmh=true) 
    {
        cur_time_ = std::chrono::steady_clock::now();
        double dt = static_cast<std::chrono::duration<double>>(cur_time_ - prev_time_).count();
 
        if (prev_img.empty() || cur_img.empty() || prev_depth.empty() || dt <= 0.0) 
            return 0.0;
        
        std::vector<cv::Point2f> pts_prev, pts_curr, pts_prev_back;
        
        // 1. Поиск фич
        cv::goodFeaturesToTrack(prev_img, pts_prev, params_.max_corners, 
                                params_.quality_level, params_.min_distance, mask);
        if (pts_prev.empty()) return 0.0;
        
        // 2. Оптический поток (Forward-Backward check)
        std::vector<uchar> status_fwd, status_bwd;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_img, cur_img, pts_prev, pts_curr, status_fwd, err);
        cv::calcOpticalFlowPyrLK(cur_img, prev_img, pts_curr, pts_prev_back, status_bwd, err);

        std::vector<cv::Point3f> object_points;
        std::vector<cv::Point2f> image_points;

        // 3. Сбор 3D-2D соответствий
        for (size_t i = 0; i < pts_prev.size(); i++) {
            if (status_fwd[i] && status_bwd[i]) {
                cv::Point2f diff = pts_prev[i] - pts_prev_back[i];
                if (diff.x*diff.x + diff.y*diff.y < 1.0f) { // FB-error threshold
                    float z = getSubpixelDepth_(prev_depth, pts_prev[i]);
                    if (z > params_.min_depth && z < params_.max_depth) {
                        float x = (pts_prev[i].x - cx_) * z / fx_;
                        float y = (pts_prev[i].y - cy_) * z / fy_;
                        object_points.emplace_back(x, y, z);
                        image_points.push_back(pts_curr[i]);
                    }
                }
            }
        }

        if (object_points.size() < static_cast<size_t>(params_.min_pnp_points)) return 0.0;

        // 4. RANSAC PnP
        cv::Mat rvec, tvec, inliers;
        bool success = cv::solvePnPRansac(object_points, image_points, K_, cv::Mat(), 
                                          rvec, tvec, false, params_.ransac_iter, 
                                          static_cast<float>(params_.ransac_reproj_error), 
                                          params_.ransac_conf, inliers);

        if (!success) return 0.0;

        float inlier_ratio = static_cast<float>(inliers.rows) / object_points.size();
        if (inlier_ratio < params_.inlier_ratio_threshold || inliers.rows < params_.min_inliers) {
            return 0.0;
        }

        // 5. Уточнение только по инлайерам
        std::vector<cv::Point3f> inlier_obj;
        std::vector<cv::Point2f> inlier_img;
        for (int i = 0; i < inliers.rows; i++) {
            int idx = inliers.at<int>(i);
            inlier_obj.push_back(object_points[idx]);
            inlier_img.push_back(image_points[idx]);
        }
        cv::solvePnPRefineLM(inlier_obj, inlier_img, K_, cv::Mat(), rvec, tvec);

        vel_ms_ = cv::norm(tvec) / dt;

        prev_time_=cur_time_;

        return to_kmh ? vel_ms_*3.6 : vel_ms_;
    }

    double GetSmoothed(bool to_kmh=true){
        smoothed_velocity_ = params_.smoothed_alpha * vel_ms_ + (1.0 - params_.smoothed_alpha) * smoothed_velocity_;
        return to_kmh ? smoothed_velocity_*3.6 : smoothed_velocity_;
    }

    void Start(){
        prev_time_=std::chrono::steady_clock::now();
        return ;
    }

    VelocityTrackerParams& getParams(){
        return params_;
    }
};
