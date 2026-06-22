#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include "stereo_params.hpp"

class IStereoMatcher {
public:
    virtual ~IStereoMatcher() = default;

    virtual void create() = 0;
    virtual void rectify(const cv::Mat &l_raw, const cv::Mat &r_raw, cv::Mat &l_rect, cv::Mat &r_rect) = 0;
    virtual cv::Mat compute(const cv::Mat &img_l_rect_in, const cv::Mat &img_r_rect_in) = 0;
    virtual IStereoMatcher& load(const std::string &filename) = 0;

    virtual const cv::Mat& getK1() const = 0;
    virtual const cv::Mat& getD1() const = 0;
    virtual const cv::Mat& getQ() const = 0;
    virtual cv::Mat getDepthMap(const cv::Mat& disp_float) = 0;
    virtual cv::Mat getMatrixLeft() = 0;
    
    virtual StereoParams& getParams() = 0;
};
