#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

class ROIManager{
    private:
        int x_, y_, width_, height_;
        cv::Size max_size_, min_size_;
        bool initialized_constraint_;

        void ApplyConstraintSize_(){
            width_ = std::clamp(width_, min_size_.width, max_size_.width);
            height_ = std::clamp(height_, min_size_.height, max_size_.height);

            x_ = std::clamp(x_, width_/2, max_size_.width - (width_ - width_/2));
            y_ = std::clamp(y_, height_/2, max_size_.height - (height_ - height_/2));
            return ;
        }
    public:
        ROIManager(cv::Size min_size=cv::Size(), cv::Size max_size=cv::Size(), int x=0, int y=0, int width=250, int height=250) 
                : min_size_(min_size), max_size_(max_size), x_(x), y_(y), width_(width), height_(height) {
                    initialized_constraint_ = (max_size_.width!=0 && max_size_.height!=0);
                }
        ~ROIManager(){}

        void moveROI(int dx, int dy) {
            x_ += dx;
            y_ += dy;
        }

        void resizeROI(int dwidth, int dheight) {
            width_ += dwidth;
            height_ += dheight;
        }

        cv::Rect GetRect() {
            ApplyConstraintSize_();
            return cv::Rect(x_ - width_/2, y_ - height_/2, width_, height_);
        }

        void SetConstraint(cv::Size min_size, cv::Size max_size){
            max_size_=max_size;
            min_size_=min_size;
            initialized_constraint_=true;
        }

        bool CheckInitConstraint(){
            return initialized_constraint_;
        }
};