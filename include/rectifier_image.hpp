#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>

#include "params_manager.hpp"

struct RectifierImageParams {
    std::string &path_out;
    std::string &path_left_imgs_directory;
    std::string &path_right_imgs_directory;
    std::string &extension;
    int &num_test_img;

    RectifierImageParams() :
        path_out(ParamsManager::getInstance()["path"]["path_out"]),
        num_test_img(ParamsManager::getInstance()["path"]["num_test_img"]),
        path_left_imgs_directory(ParamsManager::getInstance()["path"]["path_left_imgs_directory"]),
        path_right_imgs_directory(ParamsManager::getInstance()["path"]["path_right_imgs_directory"]),
        extension(ParamsManager::getInstance()["path"]["extension"]) {}

    std::string getLeftInFilename() const {
        return path_left_imgs_directory + "left" + std::to_string(num_test_img) + "." + extension;
    }

    std::string getRightInFilename() const {
        return path_right_imgs_directory + "right" + std::to_string(num_test_img) + "." + extension;
    }

    std::string getLeftOutFilename() const {
        return path_out + "left_rect." + extension;
    }

    std::string getRightOutFilename() const {
        return path_out + "right_rect." + extension;
    }
};


class RectifierImage{
	private:
		cv::Mat R1_, R2_, P1_, P2_, Q_;
		cv::Mat K1_, K2_, R_;
		cv::Vec3d T_;
		cv::Mat D1_, D2_;
		
		RectifierImageParams params_;
	public:
		RectifierImage(const std::string& filename_calib_stereo="../result/cam_stereo.yml") {
			cv::FileStorage fs(filename_calib_stereo, cv::FileStorage::READ);

			fs["K1"] >> K1_;
			fs["K2"] >> K2_;
			fs["D1"] >> D1_;
			fs["D2"] >> D2_;
			fs["R"] >> R_;
			fs["T"] >> T_;

			fs["R1"] >> R1_;
			fs["R2"] >> R2_;
			fs["P1"] >> P1_;
			fs["P2"] >> P2_;
			fs["Q"] >> Q_;
		}
		~RectifierImage(){}

		void UndistortRectify(){
			cv::Mat lmapx, lmapy, rmapx, rmapy;
			cv::Mat imgU1, imgU2;

			cv::Mat img1 = cv::imread(params_.getLeftInFilename(), cv::IMREAD_COLOR);
  			cv::Mat img2 = cv::imread(params_.getRightInFilename(), cv::IMREAD_COLOR);

			initUndistortRectifyMap(K1_, D1_, R1_, P1_, img1.size(), CV_32F, lmapx, lmapy);
			initUndistortRectifyMap(K2_, D2_, R2_, P2_, img2.size(), CV_32F, rmapx, rmapy);
			remap(img1, imgU1, lmapx, lmapy, cv::INTER_LINEAR);
			remap(img2, imgU2, rmapx, rmapy, cv::INTER_LINEAR);

			imwrite(params_.getLeftOutFilename(), imgU1);
			imwrite(params_.getRightOutFilename(), imgU2);

			return ;
		}

		RectifierImageParams& getParams(){
			return params_;
		}
};