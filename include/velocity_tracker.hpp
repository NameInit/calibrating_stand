#pragma once


#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>


class VelocityTracker{
    private:
        std::vector<cv::Point2f> points_;

    public:
        VelocityTracker(){};
        ~VelocityTracker(){};

        // VelocityTracker& bound(cv::Mat &prev_gray){

        //     return *this;
        // }
        double ComputeVelocity(const cv::Mat &prev_left, const cv::Mat &prev_right, 
            const cv::Mat &cur_left, const cv::Mat &cur_right,
            const int window_size = 15, const int search_range = 64){
            // cv::
            return {};
        }
};