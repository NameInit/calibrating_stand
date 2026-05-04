#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

struct ROIManagerParams{
    cv::Size min_size=cv::Size(); 
    cv::Size max_size=cv::Size(); 
    int x;
    int y;
    int width;
    int height;

    ROIManagerParams(const std::string& config_name = "../.config/params.yml"){
        cv::FileStorage fs(config_name, cv::FileStorage::READ);
        cv::FileNode node = fs["roi"];

        node["x"] >> x;
        node["y"] >> y;
        node["width"] >> width;
        node["height"] >> height;
    }
};

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
        ROIManager(const ROIManagerParams& p = ROIManagerParams()) 
                : min_size_(p.min_size), max_size_(p.max_size), x_(p.x), y_(p.y), width_(p.width), height_(p.height) {
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