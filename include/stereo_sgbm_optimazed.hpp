#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <string>
#include <iostream>
#include <future>

#include "params_manager.hpp"

struct StereoSGBMOptimizedParamsCreate {
    int &min_disp;
    int &max_disp;
    int &block_size;
    int &uniqueness_ratio;
    int &speckle_ws;
    int &speckle_range;
    bool &use_wls;
    int &wls_lambda;
    double &wls_sigma;
    int &median_blur_size;

    StereoSGBMOptimizedParamsCreate() :
        min_disp(ParamsManager::getInstance()["stereo_sgbm"]["min_disp"]),
        max_disp(ParamsManager::getInstance()["stereo_sgbm"]["max_disp"]),
        block_size(ParamsManager::getInstance()["stereo_sgbm"]["block_size"]),
        uniqueness_ratio(ParamsManager::getInstance()["stereo_sgbm"]["uniqueness_ratio"]),
        speckle_ws(ParamsManager::getInstance()["stereo_sgbm"]["speckle_ws"]),
        speckle_range(ParamsManager::getInstance()["stereo_sgbm"]["speckle_range"]),
        use_wls(ParamsManager::getInstance()["stereo_sgbm"]["use_wls"]),
        wls_lambda(ParamsManager::getInstance()["stereo_sgbm"]["wls_lambda"]),
        wls_sigma(ParamsManager::getInstance()["stereo_sgbm"]["wls_sigma"]),
        median_blur_size(ParamsManager::getInstance()["stereo_sgbm"]["median_blur_size"]) {}

    int getP1() const { return 8 * 3 * block_size * block_size; }
    int getP2() const { return 32 * 3 * block_size * block_size; }
};

class StereoSGBMOptimized {
private:
    cv::Mat K1_, D1_, K2_, D2_, R_, R1_, R2_, P1_, P2_, Q_, E_, F_;
    cv::Vec3d T_; 
    cv::Mat map11_, map12_, map21_, map22_;

    cv::Ptr<cv::StereoSGBM> stereo_bm_left_;
    cv::Ptr<cv::StereoMatcher> stereo_bm_right_;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter_;
    cv::Ptr<cv::CLAHE> clahe_;
    
    StereoSGBMOptimizedParamsCreate params_;
    cv::Size current_img_size_ = cv::Size(0, 0);

    const double scale_factor = 0.75; 

public:
    StereoSGBMOptimized() {}
    StereoSGBMOptimized(const std::string &filename = "../result/cam_stereo.yml") {
        load(filename);
    }

    const cv::Mat& getK1() const { return K1_; }
    const cv::Mat& getD1() const { return D1_; }
    const cv::Mat& getQ()  const { return Q_; }

    void create() {
        int effective_max_disp = (int(params_.max_disp * scale_factor) / 16) * 16;
        if (effective_max_disp < 16) effective_max_disp = 16;

        stereo_bm_left_ = cv::StereoSGBM::create(params_.min_disp, effective_max_disp, params_.block_size, 
                                                params_.getP1(), params_.getP2(), 
                                                1, 63, params_.uniqueness_ratio, params_.speckle_ws, 
                                                params_.speckle_range, cv::StereoSGBM::MODE_SGBM_3WAY);

        if (params_.use_wls) {
            stereo_bm_right_ = cv::ximgproc::createRightMatcher(stereo_bm_left_);
            wls_filter_ = cv::ximgproc::createDisparityWLSFilter(stereo_bm_left_);
            wls_filter_->setLambda(params_.wls_lambda);
            wls_filter_->setSigmaColor(params_.wls_sigma);
        }

        clahe_ = cv::createCLAHE(2.0, cv::Size(8, 8));
    }

    void rectify(const cv::Mat &l_raw, const cv::Mat &r_raw, cv::Mat &l_rect, cv::Mat &r_rect) {
        if (map11_.empty() || l_raw.size() != current_img_size_) {
            current_img_size_ = l_raw.size();
            cv::initUndistortRectifyMap(K1_, D1_, R1_, P1_, current_img_size_, CV_16SC2, map11_, map12_);
            cv::initUndistortRectifyMap(K2_, D2_, R2_, P2_, current_img_size_, CV_16SC2, map21_, map22_);
        }
        cv::remap(l_raw, l_rect, map11_, map12_, cv::INTER_LINEAR);
        cv::remap(r_raw, r_rect, map21_, map22_, cv::INTER_LINEAR);
    }

    cv::Mat compute(const cv::Mat &img_l_rect_in, const cv::Mat &img_r_rect_in) {
        cv::Mat img_l_rect, img_r_rect;
        
        auto fut_l = std::async(std::launch::async, [&](){ clahe_->apply(img_l_rect_in, img_l_rect); });
        auto fut_r = std::async(std::launch::async, [&](){ clahe_->apply(img_r_rect_in, img_r_rect); });
        fut_l.get(); fut_r.get();

        cv::Mat l_small, r_small;
        cv::resize(img_l_rect, l_small, cv::Size(), scale_factor, scale_factor, cv::INTER_AREA);
        cv::resize(img_r_rect, r_small, cv::Size(), scale_factor, scale_factor, cv::INTER_AREA);

        cv::Mat disp_left, disp_right, filtered_disp;

        if (params_.use_wls && !stereo_bm_right_.empty()) {
            auto fut_match_r = std::async(std::launch::async, [&](){
                stereo_bm_right_->compute(r_small, l_small, disp_right);
            });
            stereo_bm_left_->compute(l_small, r_small, disp_left);
            fut_match_r.get();

            wls_filter_->filter(disp_left, l_small, filtered_disp, disp_right);
        } else {
            stereo_bm_left_->compute(l_small, r_small, disp_left);
            filtered_disp = disp_left;
        }

        cv::Mat final_disp;
        cv::resize(filtered_disp, final_disp, img_l_rect_in.size(), 0, 0, cv::INTER_LINEAR);
        final_disp *= (1.0 / scale_factor);

        if (params_.median_blur_size > 0) cv::medianBlur(final_disp, final_disp, params_.median_blur_size);

        cv::Mat disp_float;
        final_disp.convertTo(disp_float, CV_32F, 1.0 / 16.0);
        return disp_float;
    }

    StereoSGBMOptimized& load(const std::string &filename = "../result/cam_stereo.yml"){
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened()) throw std::runtime_error("Cannot open: " + filename);
        fs["K1"] >> K1_; fs["D1"] >> D1_;
        fs["K2"] >> K2_; fs["D2"] >> D2_;
        fs["R"]  >> R_;  fs["T"]  >> T_;
        fs["R1"] >> R1_; fs["R2"] >> R2_;
        fs["P1"] >> P1_; fs["P2"] >> P2_;
        fs["Q"]  >> Q_;  fs["E"]  >> E_;
        fs["F"]  >> F_;
        fs.release();
        return *this;
    }

    cv::Mat getDepthMap(const cv::Mat& disp_float) {
        if (disp_float.empty()) return cv::Mat();
        cv::Mat image_3d, depth_map, channels[3];
        cv::reprojectImageTo3D(disp_float, image_3d, Q_);
        cv::split(image_3d, channels);
        depth_map = channels[2];
        depth_map.setTo(0, disp_float <= 0);
        return depth_map;
    }

    cv::Mat getMatrixLeft(){ return K1_; }
    StereoSGBMOptimizedParamsCreate& getParams(){ return params_; }
};