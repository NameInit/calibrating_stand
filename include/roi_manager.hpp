#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

#include "params_manager.hpp"

struct ROIManagerParams{
    int &x;
    int &y;
    int &width;
    int &height;

    ROIManagerParams() :
        x(ParamsManager::getInstance()["roi"]["x"]),
        y(ParamsManager::getInstance()["roi"]["y"]),
        width(ParamsManager::getInstance()["roi"]["width"]),
        height(ParamsManager::getInstance()["roi"]["height"]) {}
};

class ROIManager{
    private:
        ROIManagerParams params_;
        cv::Size max_size_, min_size_;
        bool initialized_constraint_;

        void ApplyConstraintSize_(){
            params_.width = std::clamp(params_.width, min_size_.width, max_size_.width);
            params_.height = std::clamp(params_.height, min_size_.height, max_size_.height);

            params_.x = std::clamp(params_.x, params_.width/2, max_size_.width - (params_.width - params_.width/2));
            params_.y = std::clamp(params_.y, params_.height/2, max_size_.height - (params_.height - params_.height/2));
            return ;
        }
    public:
        ROIManager(){}
        ~ROIManager(){}

        void moveROI(int dx, int dy) {
            params_.x += dx;
            params_.y += dy;
        }

        void resizeROI(int dwidth, int dheight) {
            params_.width += dwidth;
            params_.height += dheight;
        }

        cv::Rect GetRect() {
            ApplyConstraintSize_();
            return cv::Rect(params_.x - params_.width/2, params_.y - params_.height/2, params_.width, params_.height);
        }

        void SetConstraint(cv::Size min_size, cv::Size max_size){
            max_size_=max_size;
            min_size_=min_size;
            initialized_constraint_=true;
        }

        bool CheckInitConstraint(){
            return initialized_constraint_;
        }

        ROIManagerParams& getParams(){
            return params_;
        }
};