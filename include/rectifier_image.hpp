#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>


struct RectifierFileParams{
	std::string leftin_filename;
	std::string rightin_filename;
	std::string leftout_filename;
	std::string rightout_filename;
	std::string path_left_imgs_directory;
	std::string path_right_imgs_directory;
	std::string extension;
	std::string path_out;
	int num_test_img;

	RectifierFileParams(const std::string& config_name = "../.config/params.yml"){
		cv::FileStorage fs(config_name, cv::FileStorage::READ);
        cv::FileNode node = fs["path"];

		node["path_out"] >> path_out;
		node["num_test_img"] >> num_test_img;
		node["path_left_imgs_directory"] >> path_left_imgs_directory;
		node["path_right_imgs_directory"] >> path_right_imgs_directory;
		node["extension"] >> extension;
		leftin_filename = path_left_imgs_directory+"left"+std::to_string(num_test_img)+"."+extension;
		rightin_filename = path_right_imgs_directory+"right"+std::to_string(num_test_img)+"."+extension;
		leftout_filename = path_out+"left_rect"+"."+extension;
		rightout_filename = path_out+"right_rect"+"."+extension;
	}
};

class RectifierImage{
	public:
		RectifierImage(const std::string& filename_calib_stereo="../result/cam_stereo.yml")
		: fs1_(filename_calib_stereo, cv::FileStorage::READ) {
			fs1_["K1"] >> K1_;
			fs1_["K2"] >> K2_;
			fs1_["D1"] >> D1_;
			fs1_["D2"] >> D2_;
			fs1_["R"] >> R_;
			fs1_["T"] >> T_;

			fs1_["R1"] >> R1_;
			fs1_["R2"] >> R2_;
			fs1_["P1"] >> P1_;
			fs1_["P2"] >> P2_;
			fs1_["Q"] >> Q_;
		}
		~RectifierImage(){}

		void UndistortRectify(const RectifierFileParams& p = RectifierFileParams()){
			cv::Mat lmapx, lmapy, rmapx, rmapy;
			cv::Mat imgU1, imgU2;

			cv::Mat img1 = cv::imread(p.leftin_filename, cv::IMREAD_COLOR);
  			cv::Mat img2 = cv::imread(p.rightin_filename, cv::IMREAD_COLOR);

			initUndistortRectifyMap(K1_, D1_, R1_, P1_, img1.size(), CV_32F, lmapx, lmapy);
			initUndistortRectifyMap(K2_, D2_, R2_, P2_, img2.size(), CV_32F, rmapx, rmapy);
			remap(img1, imgU1, lmapx, lmapy, cv::INTER_LINEAR);
			remap(img2, imgU2, rmapx, rmapy, cv::INTER_LINEAR);

			imwrite(p.leftout_filename, imgU1);
			imwrite(p.rightout_filename, imgU2);

			return ;
		}
	private:
		cv::Mat R1_, R2_, P1_, P2_, Q_;
		cv::Mat K1_, K2_, R_;
		cv::Vec3d T_;
		cv::Mat D1_, D2_;
		cv::FileStorage fs1_;
};