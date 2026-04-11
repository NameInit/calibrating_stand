#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <string>
#include <vector>
#include <iostream>

class StereoSGBM{
    private:
        cv::Mat K1_, D1_, K2_, D2_, R_, R1_, R2_, P1_, P2_, Q_, E_, F_;
        cv::Vec3d T_; 
        cv::Mat map11_, map12_, map21_, map22_;

        cv::Ptr<cv::StereoSGBM> stereo_bm_left_;
        cv::Ptr<cv::StereoMatcher> stereo_bm_right_;
        cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter_;

        bool use_wls_filter_;
        int median_blur_ksize_;
    public:
        StereoSGBM(const std::string &filename="../result/cam_stereo.yml"){
            cv::FileStorage fs(filename, cv::FileStorage::READ);

            if (!fs.isOpened()) {
                throw std::runtime_error("Cannot open: " + filename);
            }

            fs["K1"] >> K1_;
            fs["D1"] >> D1_;
            fs["K2"] >> K2_;
            fs["D2"] >> D2_;
            fs["R"]  >> R_;
            fs["T"]  >> T_;
            fs["R1"] >> R1_;
            fs["R2"] >> R2_;
            fs["P1"] >> P1_;
            fs["P2"] >> P2_;
            fs["Q"]  >> Q_;
            fs["E"]  >> E_;
            fs["F"]  >> F_;

            fs.release();
            return;
        }
        ~StereoSGBM(){}

    void Create(int min_disp, int num_disp, int block_size, int p1, int p2, 
                int uniqueness_ratio, int speckle_ws, int speckle_range,
                bool use_wls, int wls_lambda, float wls_sigma, int median_blur_size){
        
        use_wls_filter_ = use_wls;

        stereo_bm_left_ = cv::StereoSGBM::create(
            min_disp,
            num_disp,
            block_size,
            p1,
            p2,
            2,                      // disp12MaxDiff
            63,                     // preFilterCap
            uniqueness_ratio,
            speckle_ws,
            speckle_range,
            cv::StereoSGBM::MODE_SGBM_3WAY
        );

        if (use_wls_filter_) {
            stereo_bm_right_ = cv::ximgproc::createRightMatcher(stereo_bm_left_);
            wls_filter_ = cv::ximgproc::createDisparityWLSFilter(stereo_bm_left_);
            wls_filter_->setLambda(wls_lambda);
            wls_filter_->setSigmaColor(wls_sigma);
        }

        median_blur_ksize_ = (median_blur_size > 0 && median_blur_size % 2 == 0) 
                             ? median_blur_size + 1 
                             : median_blur_size;
        return;
    }

    void InitRectification(cv::Size img_size) {
        cv::initUndistortRectifyMap(K1_, D1_, R1_, P1_, img_size, CV_16SC2, map11_, map12_);
        cv::initUndistortRectifyMap(K2_, D2_, R2_, P2_, img_size, CV_16SC2, map21_, map22_);
    }

    cv::Mat Compute(const cv::Mat &img_left_raw, const cv::Mat &img_right_raw) {
        cv::Mat img_l, img_r;
        
        cv::remap(img_left_raw, img_l, map11_, map12_, cv::INTER_LINEAR);
        cv::remap(img_right_raw, img_r, map21_, map22_, cv::INTER_LINEAR);

        cv::Mat disp_left, disp_right;
        cv::Mat filtered_disp;

        stereo_bm_left_->compute(img_l, img_r, disp_left);

        if (use_wls_filter_) {
            stereo_bm_right_->compute(img_r, img_l, disp_right);

            wls_filter_->filter(disp_left, img_l, filtered_disp, disp_right);
        } else {
            filtered_disp = disp_left;
        }

        if (median_blur_ksize_ > 0) {
            cv::medianBlur(filtered_disp, filtered_disp, median_blur_ksize_);
        }

        cv::Mat disp_float;
        filtered_disp.convertTo(disp_float, CV_32F, 1.0 / 16.0);

        return disp_float;
    }

    cv::Mat GetDepthMap(const cv::Mat& disp_float) {
        if (disp_float.empty()) return cv::Mat();

        cv::Mat image_3d;
        cv::reprojectImageTo3D(disp_float, image_3d, Q_);

        cv::Mat depth_map;
        cv::Mat channels[3];
        cv::split(image_3d, channels);
        depth_map = channels[2];

        depth_map.setTo(0, disp_float <= 0);
        
        return depth_map;
    }
};