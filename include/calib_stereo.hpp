#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <filesystem>
#include <cstring>
#include <iostream>

#include "params_manager.hpp"

struct CalibratingStereoParams{
	std::string &filename_left_cam_params;
	std::string &filename_right_cam_params;
	std::string &path_params;

	CalibratingStereoParams(const std::string& config_name = "../.config/params.yml") :
		filename_left_cam_params(ParamsManager::getInstance()["path"]["filename_left_cam_params"]),
		filename_right_cam_params(ParamsManager::getInstance()["path"]["filename_right_cam_params"]),
		path_params(ParamsManager::getInstance()["path"]["out_params"]) {}

	std::string getLeftCalibFile() const {
		return path_params+filename_left_cam_params;
	}

	std::string getRightCalibFile() const {
		return path_params+filename_right_cam_params;
	}
};

struct StereoDatasetParams{
	std::string &leftimg_dir;
	std::string &rightimg_dir;
	std::string &leftimg_filename;
	std::string &rightimg_filename;
	std::string &extension;

	StereoDatasetParams(const std::string& config_name = "../.config/params.yml") :
		leftimg_dir(ParamsManager::getInstance()["path"]["left_imgs_directory"]),
		rightimg_dir(ParamsManager::getInstance()["path"]["right_imgs_directory"]),
		leftimg_filename(ParamsManager::getInstance()["path"]["img_left_filename"]),
		rightimg_filename(ParamsManager::getInstance()["path"]["img_right_filename"]),
		extension(ParamsManager::getInstance()["path"]["extension"]) {}
};

class CalibratingStereo{
	private:
		std::vector<std::vector<cv::Point3f>> object_points_;
		std::vector<std::vector<cv::Point2f>> imagePoints1_, imagePoints2_;
		std::vector<cv::Point2f> corners1_, corners2_;
		std::vector<std::vector<cv::Point2f>> left_img_points_, right_img_points_;

		cv::Mat img1_, img2_, gray1_, gray2_;

		cv::FileStorage fsl_, fsr_;
		
		cv::Mat K1_, K2_, R_, F_, E_;
		cv::Vec3d T_;
		cv::Mat D1_, D2_;
		cv::Mat R1_, R2_, P1_, P2_, Q_;

		int board_width_, board_height_;
		float square_size_;

		bool DoesExist_(const std::string& name) {
			if(!std::filesystem::exists(name) || !std::filesystem::is_regular_file(name)){
				return false;
			}
			return true;
		}
	public:
		CalibratingStereo(const CalibratingStereoParams& p = CalibratingStereoParams()) 
		: fsl_(p.getLeftCalibFile(), cv::FileStorage::READ), fsr_(p.getRightCalibFile(), cv::FileStorage::READ){
			if (!fsl_.isOpened()) {
				std::cerr << "Error: Cannot open left calibration file: " << p.getLeftCalibFile() << std::endl;
				exit(1);
			}
			if (!fsr_.isOpened()) {
				std::cerr << "Error: Cannot open right calibration file: " << p.getRightCalibFile() << std::endl;
				exit(1);
			}
			
			fsl_["K"] >> K1_;
			fsr_["K"] >> K2_;
			fsl_["D"] >> D1_;
			fsr_["D"] >> D2_;
			fsl_["board_width"] >> board_width_;
			fsl_["board_height"] >> board_height_;
			fsl_["square_size"] >> square_size_;
			
			if (K1_.empty() || K2_.empty() || D1_.empty() || D2_.empty()) {
				std::cerr << "Error: Failed to load camera matrices or distortion coefficients" << std::endl;
				exit(1);
			}
			if (board_width_ <= 0 || board_height_ <= 0 || square_size_ <= 0) {
				std::cerr << "Error: Invalid board dimensions or square size" << std::endl;
				exit(1);
			}
		}
		
		~CalibratingStereo(){}

		void LoadImagePoints(const StereoDatasetParams& p = StereoDatasetParams()) {
			cv::Size board_size = cv::Size(board_width_, board_height_);
			int board_n = board_width_ * board_height_;

			for (int i = 1; ; i++) {
				std::string left_img, right_img;
				left_img = p.leftimg_dir + p.leftimg_filename + std::to_string(i) + "." + p.extension;
				right_img = p.rightimg_dir + p.rightimg_filename + std::to_string(i) + "." + p.extension;
				
				if(!DoesExist_(left_img)||!DoesExist_(right_img)){
					break;
				}

				img1_ = cv::imread(left_img, cv::IMREAD_COLOR);
				img2_ = cv::imread(right_img, cv::IMREAD_COLOR);
				
				if (img1_.empty()) {
					std::cerr << "Warning: Cannot read left image: " << left_img << std::endl;
					continue;
				}
				if (img2_.empty()) {
					std::cerr << "Warning: Cannot read right image: " << right_img << std::endl;
					continue;
				}
				
				if (img1_.size() != img2_.size()) {
					std::cerr << "Warning: Image size mismatch for pair " << i << std::endl;
					continue;
				}
				
				cv::cvtColor(img1_, gray1_, cv::COLOR_BGR2GRAY);
				cv::cvtColor(img2_, gray2_, cv::COLOR_BGR2GRAY);

				bool found1 = false, found2 = false;

				found1 = cv::findChessboardCorners(img1_, board_size, corners1_,
			cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);
				found2 = cv::findChessboardCorners(img2_, board_size, corners2_,
			cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);

				if(!found1 || !found2){
					std::cout << "Chessboard find error!" << std::endl;
					std::cout << "leftImg: " << left_img << " and rightImg: " << right_img <<std::endl;
					continue;
				} 

				if (found1) {
					cv::cornerSubPix(gray1_, corners1_, cv::Size(5, 5), cv::Size(-1, -1),
						cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.1));
					cv::drawChessboardCorners(gray1_, board_size, corners1_, found1);
				}
				
				if (found2) {
					cv::cornerSubPix(gray2_, corners2_, cv::Size(5, 5), cv::Size(-1, -1),
						cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.1));
					cv::drawChessboardCorners(gray2_, board_size, corners2_, found2);
				}

				if (corners1_.size() != corners2_.size()) {
					std::cerr << "Warning: Different number of corners detected in pair " << i << std::endl;
					continue;
				}

				std::vector< cv::Point3f > obj;
				for (int i = 0; i < board_height_; i++)
					for (int j = 0; j < board_width_; j++)
						obj.push_back(cv::Point3f((float)j * square_size_, (float)i * square_size_, 0));

				if (found1 && found2) {
					std::cout << i << ". Found corners!" << std::endl;
					imagePoints1_.push_back(corners1_);
					imagePoints2_.push_back(corners2_);
					object_points_.push_back(obj);
				}
			}
			
			if (imagePoints1_.empty()) {
				std::cerr << "Error: No valid image pairs found" << std::endl;
				return;
			}
			
			for (int i = 0; i < imagePoints1_.size(); i++) {
				std::vector< cv::Point2f > v1, v2;
				for (int j = 0; j < imagePoints1_[i].size(); j++) {
					v1.push_back(cv::Point2f((double)imagePoints1_[i][j].x, (double)imagePoints1_[i][j].y));
					v2.push_back(cv::Point2f((double)imagePoints2_[i][j].x, (double)imagePoints2_[i][j].y));
				}
				left_img_points_.push_back(v1);
				right_img_points_.push_back(v2);
			}
		}

		void ComputeMatrixStereo(){
			if (object_points_.empty() || left_img_points_.empty() || right_img_points_.empty()) {
				std::cerr << "Error: No image points loaded. Call LoadImagePoints first." << std::endl;
				return;
			}
			
			if (img1_.empty()) {
				std::cerr << "Error: No image size available for calibration" << std::endl;
				return;
			}
			
			double rms = stereoCalibrate(object_points_, left_img_points_, right_img_points_, 
			                              K1_, D1_, K2_, D2_, img1_.size(), R_, T_, E_, F_,
			                              cv::CALIB_FIX_INTRINSIC,
			                              cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 100, 1e-5));
			
			std::cout << "Stereo calibration RMS error: " << rms << std::endl;
			
			if (rms > 1.0) {
				std::cerr << "Warning: High calibration RMS error. Results may be inaccurate." << std::endl;
			}
		}

		void ComputeMatrixStereoRectify(){
			if (R_.empty()) {
				std::cerr << "Error: Stereo calibration not computed. Call ComputeMatrixStereo first." << std::endl;
				return;
			}
			
			if (img1_.empty()) {
				std::cerr << "Error: No image size available for rectification" << std::endl;
				return;
			}
			
			stereoRectify(K1_, D1_, K2_, D2_, img1_.size(), R_, T_, R1_, R2_, P1_, P2_, Q_);
		}

		void SaveMatrixTo(const std::string& filename_out = "result_stereo.yml"){
			cv::FileStorage fs1(filename_out, cv::FileStorage::WRITE);
			if (!fs1.isOpened()) {
				std::cerr << "Error: Cannot open output file for writing: " << filename_out << std::endl;
				return;
			}
			
			fs1 << "K1" << K1_;
			fs1 << "K2" << K2_;
			fs1 << "D1" << D1_;
			fs1 << "D2" << D2_;
			fs1 << "R" << R_;
			fs1 << "T" << T_;
			fs1 << "E" << E_;
			fs1 << "F" << F_;
			fs1 << "R1" << R1_;
			fs1 << "R2" << R2_;
			fs1 << "P1" << P1_;
			fs1 << "P2" << P2_;
			fs1 << "Q" << Q_;
			
			std::cout << "Stereo calibration saved to: " << filename_out << std::endl;
		}
};