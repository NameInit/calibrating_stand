#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <sys/stat.h>


class CalibratingCam{
	public:
		CalibratingCam(){}
		~CalibratingCam(){}

		bool SetupCalibration(int board_width, int board_height, int num_imgs, 
		                      float square_size,const char* imgs_directory, 
		                      const char* imgs_filename, const char* extension) {
			cv::Size board_size = cv::Size(board_width, board_height);
			
			board_width_ = board_width;
			board_height_ = board_height;
			square_size_ = square_size;

			bool found_any = false;
			cv::Size img_size;
			for (int k = 1; k <= num_imgs; k++) {
				char img_file[256];
				sprintf(img_file, "%s%s%d.%s", imgs_directory, imgs_filename, k, extension);

				if(!DoesExist_(img_file))
					continue;

				cv::Mat img = cv::imread(img_file, cv::IMREAD_COLOR);
				if(img.empty()) {
					std::cout << "Warning: Could not read image " << img_file << std::endl;
					continue;
				}
				
				if(img_size.empty())
					img_size = img.size();
					
				cv::Mat gray;
				cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

				std::vector<cv::Point2f> corners;
				bool found = cv::findChessboardCorners(img, board_size, corners,
				                    cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);
				                    
				if (found) {
					cornerSubPix(gray, corners, cv::Size(5, 5), cv::Size(-1, -1),
					           cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.1));
					cv::drawChessboardCorners(gray, board_size, corners, found);
					
					std::vector<cv::Point3f> obj;
					for (int i = 0; i < board_height; i++)
						for (int j = 0; j < board_width; j++)
							obj.push_back(cv::Point3f((float)j * square_size, 
							                      (float)i * square_size, 0));

					std::cout << k << ". Found corners!" << std::endl;
					image_points_.push_back(corners);
					object_points_.push_back(obj);
					found_any = true;
				}
			}
			
			if(!found_any) {
				std::cout << "Error: No chessboard corners found in any image!" << std::endl;
				return false;
			}
			
			img_size_ = img_size;
			return true;
		}
		
		bool ComputeMatrixCalibration(int flag = cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5) {
			if(object_points_.empty() || image_points_.empty()) {
				std::cout << "Error: No calibration data available. Run SetupCalibration first." << std::endl;
				return false;
			}
			
			if(img_size_.empty()) {
				std::cout << "Error: Image size not set." << std::endl;
				return false;
			}
			
			cv::calibrateCamera(object_points_, image_points_, img_size_, K_, D_, rvecs_, tvecs_, flag);
			return true;
		}

		double ComputeReprojectionErrors() {
			if(object_points_.empty() || K_.empty()) {
				std::cout << "Error: No calibration data or camera matrix available." << std::endl;
				return -1.0;
			}
			
			std::vector<cv::Point2f> imagePoints2;
			int totalPoints = 0;
			double totalErr = 0;
			
			for (size_t i = 0; i < object_points_.size(); ++i) {
				projectPoints(cv::Mat(object_points_[i]), rvecs_[i], tvecs_[i], 
				              K_, D_, imagePoints2);
				double err = cv::norm(cv::Mat(image_points_[i]), cv::Mat(imagePoints2), cv::NORM_L2);
				int n = (int)object_points_[i].size();
				totalErr += err * err;
				totalPoints += n;
			}
			return std::sqrt(totalErr / totalPoints);
		}

		void SaveMatrixTo(const std::string& filename_out = "result_calibrating.yml") {
			if(K_.empty()) {
				std::cout << "Warning: Camera matrix is empty. Nothing to save." << std::endl;
				return;
			}
			
			cv::FileStorage fs(filename_out, cv::FileStorage::WRITE);
			if(!fs.isOpened()) {
				std::cout << "Error: Could not open file " << filename_out << " for writing." << std::endl;
				return;
			}
			
			fs << "K" << K_;
			fs << "D" << D_;
			fs << "board_width" << board_width_;
			fs << "board_height" << board_height_;
			fs << "square_size" << square_size_;
			std::cout << "Calibration saved to " << filename_out << std::endl;
		}
		
		cv::Mat getCameraMatrix() const { return K_; }
		cv::Mat getDistCoeffs() const { return D_; }
		std::vector<cv::Mat> getRotationVectors() const { return rvecs_; }
		std::vector<cv::Mat> getTranslationVectors() const { return tvecs_; }

	private:
		bool DoesExist_(const std::string& name) {
			struct stat buffer;   
			return (stat(name.c_str(), &buffer) == 0); 
		}

		std::vector<std::vector<cv::Point3f>> object_points_;
		std::vector<std::vector<cv::Point2f>> image_points_;
		std::vector<cv::Mat> rvecs_, tvecs_;
		
		cv::Mat K_, D_;
		cv::Size img_size_;
		
		int board_width_, board_height_;
		float square_size_;
};