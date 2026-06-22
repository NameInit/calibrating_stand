#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <string>
#include <iostream>

#include "stereo_params.hpp"
#include "istereo_matcher.hpp"


class StereoSGBM : public IStereoMatcher {
private:
    cv::Mat K1_, D1_, K2_, D2_, R_, R1_, R2_, P1_, P2_, Q_, E_, F_;
    cv::Vec3d T_; 
    cv::Mat map11_, map12_, map21_, map22_;

    cv::Ptr<cv::StereoSGBM> stereo_bm_left_;
    cv::Ptr<cv::StereoMatcher> stereo_bm_right_;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter_;
    cv::Ptr<cv::CLAHE> clahe_;
    
    StereoParams params_;

    cv::Size current_img_size_ = cv::Size(0, 0);

public:
    StereoSGBM() {}
    StereoSGBM(const std::string &filename = "../result/cam_stereo.yml") {
        load(filename);
    }

    ~StereoSGBM() override = default;

    const cv::Mat& getK1() const override { return K1_; }
    const cv::Mat& getD1() const override { return D1_; }
    const cv::Mat& getQ()  const override { return Q_; }

    void create() override {
        stereo_bm_left_ = cv::StereoSGBM::create(params_.min_disp, params_.max_disp, params_.block_size, params_.getP1(), params_.getP2(), 
                                                2, 63, params_.uniqueness_ratio, params_.speckle_ws, 
                                                params_.speckle_range, cv::StereoSGBM::MODE_SGBM_3WAY);

        if (params_.use_wls) {
            stereo_bm_right_ = cv::ximgproc::createRightMatcher(stereo_bm_left_);
            wls_filter_ = cv::ximgproc::createDisparityWLSFilter(stereo_bm_left_);
            wls_filter_->setLambda(params_.wls_lambda);
            wls_filter_->setSigmaColor(params_.wls_sigma);
        }

        clahe_ = cv::createCLAHE(2.0, cv::Size(8, 8));
    }

    void rectify(const cv::Mat &l_raw, const cv::Mat &r_raw, cv::Mat &l_rect, cv::Mat &r_rect) override {
        if (map11_.empty() || l_raw.size() != current_img_size_) {
            current_img_size_ = l_raw.size();
            cv::initUndistortRectifyMap(K1_, D1_, R1_, P1_, current_img_size_, CV_16SC2, map11_, map12_);
            cv::initUndistortRectifyMap(K2_, D2_, R2_, P2_, current_img_size_, CV_16SC2, map21_, map22_);
        }

        cv::remap(l_raw, l_rect, map11_, map12_, cv::INTER_LINEAR);
        cv::remap(r_raw, r_rect, map21_, map22_, cv::INTER_LINEAR);
    }

    cv::Mat compute(const cv::Mat &img_l_rect_in, const cv::Mat &img_r_rect_in) override {
        cv::Mat img_l_rect, img_r_rect;
        clahe_->apply(img_l_rect_in, img_l_rect);
        clahe_->apply(img_r_rect_in, img_r_rect);

        cv::Mat disp_left, disp_right, filtered_disp;
        stereo_bm_left_->compute(img_l_rect, img_r_rect, disp_left);

        if (params_.use_wls && !stereo_bm_right_.empty()) {
            stereo_bm_right_->compute(img_r_rect, img_l_rect, disp_right);
            wls_filter_->filter(disp_left, img_l_rect, filtered_disp, disp_right);
        } else {
            filtered_disp = disp_left;
        }

        if (params_.median_blur_size > 0) cv::medianBlur(filtered_disp, filtered_disp, params_.median_blur_size);

        cv::Mat disp_float;
        filtered_disp.convertTo(disp_float, CV_32F, 1.0 / 16.0);
        return disp_float;
    }

    IStereoMatcher& load(const std::string &filename = "../result/cam_stereo.yml") override {
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

    cv::Mat getDepthMap(const cv::Mat& disp_float) override {
        if (disp_float.empty()) return cv::Mat();
        cv::Mat image_3d, depth_map, channels[3];
        cv::reprojectImageTo3D(disp_float, image_3d, Q_);
        cv::split(image_3d, channels);
        depth_map = channels[2];
        depth_map.setTo(0, disp_float <= 0);
        return depth_map;
    }

    cv::Mat getMatrixLeft() override {
        return K1_;
    }

    StereoParams& getParams() override {
        return params_;
    }
};
