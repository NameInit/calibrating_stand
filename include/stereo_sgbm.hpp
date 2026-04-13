#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <string>
#include <iostream>

class StereoSGBM {
private:
    cv::Mat K1_, D1_, K2_, D2_, R_, R1_, R2_, P1_, P2_, Q_, E_, F_;
    cv::Vec3d T_; 
    cv::Mat map11_, map12_, map21_, map22_;

    cv::Ptr<cv::StereoSGBM> stereo_bm_left_;
    cv::Ptr<cv::StereoMatcher> stereo_bm_right_;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter_;

    bool use_wls_filter_ = false;
    int median_blur_ksize_ = 0;

    cv::Size current_img_size_ = cv::Size(0, 0);
    cv::Ptr<cv::CLAHE> clahe_;

public:
    StereoSGBM(const std::string &filename = "../result/cam_stereo.yml") {
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
    }

    void Create(int min_disp, int num_disp, int block_size, int p1, int p2, 
                int uniqueness_ratio, int speckle_ws, int speckle_range,
                bool use_wls, int wls_lambda, float wls_sigma, int median_blur_size) {
        
        use_wls_filter_ = use_wls;
        stereo_bm_left_ = cv::StereoSGBM::create(min_disp, num_disp, block_size, p1, p2, 
                                                2, 63, uniqueness_ratio, speckle_ws, 
                                                speckle_range, cv::StereoSGBM::MODE_SGBM_3WAY);

        if (use_wls_filter_) {
            stereo_bm_right_ = cv::ximgproc::createRightMatcher(stereo_bm_left_);
            wls_filter_ = cv::ximgproc::createDisparityWLSFilter(stereo_bm_left_);
            wls_filter_->setLambda(wls_lambda);
            wls_filter_->setSigmaColor(wls_sigma);
        }

        median_blur_ksize_ = (median_blur_size > 0 && median_blur_size % 2 == 0) 
                             ? median_blur_size + 1 : median_blur_size;

        clahe_ = cv::createCLAHE(2.0, cv::Size(8, 8));
    }

    void Rectify(const cv::Mat &l_raw, const cv::Mat &r_raw, cv::Mat &l_rect, cv::Mat &r_rect) {
        if (map11_.empty() || l_raw.size() != current_img_size_) {
            current_img_size_ = l_raw.size();
            cv::initUndistortRectifyMap(K1_, D1_, R1_, P1_, current_img_size_, CV_16SC2, map11_, map12_);
            cv::initUndistortRectifyMap(K2_, D2_, R2_, P2_, current_img_size_, CV_16SC2, map21_, map22_);
        }

        cv::remap(l_raw, l_rect, map11_, map12_, cv::INTER_LINEAR);
        cv::remap(r_raw, r_rect, map21_, map22_, cv::INTER_LINEAR);
    }

    cv::Mat Compute(const cv::Mat &img_l_rect_in, const cv::Mat &img_r_rect_in) {
        cv::Mat img_l_rect, img_r_rect;
        clahe_->apply(img_l_rect_in, img_l_rect);
        clahe_->apply(img_r_rect_in, img_r_rect);

        cv::Mat disp_left, disp_right, filtered_disp;
        stereo_bm_left_->compute(img_l_rect, img_r_rect, disp_left);

        if (use_wls_filter_ && !stereo_bm_right_.empty()) {
            stereo_bm_right_->compute(img_r_rect, img_l_rect, disp_right);
            wls_filter_->filter(disp_left, img_l_rect, filtered_disp, disp_right);
        } else {
            filtered_disp = disp_left;
        }

        if (median_blur_ksize_ > 0) cv::medianBlur(filtered_disp, filtered_disp, median_blur_ksize_);

        cv::Mat disp_float;
        filtered_disp.convertTo(disp_float, CV_32F, 1.0 / 16.0);
        return disp_float;
    }

    cv::Mat GetDepthMap(const cv::Mat& disp_float) {
        if (disp_float.empty()) return cv::Mat();
        cv::Mat image_3d, depth_map, channels[3];
        cv::reprojectImageTo3D(disp_float, image_3d, Q_);
        cv::split(image_3d, channels);
        depth_map = channels[2];
        depth_map.setTo(0, disp_float <= 0);
        return depth_map;
    }
};
