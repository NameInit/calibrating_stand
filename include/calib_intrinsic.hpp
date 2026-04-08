#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <sys/stat.h>

using namespace std;
using namespace cv;

class CalibratingCam{
	public:
		CalibratingCam(){}
		~CalibratingCam(){}

		bool SetupCalibration(int board_width, int board_height, int num_imgs, 
		                      float square_size,const char* imgs_directory, 
		                      const char* imgs_filename, const char* extension) {
			Size board_size = Size(board_width, board_height);
			
			board_width_ = board_width;
			board_height_ = board_height;
			square_size_ = square_size;

			bool found_any = false;
			Size img_size;
			for (int k = 1; k <= num_imgs; k++) {
				char img_file[256];
				sprintf(img_file, "%s%s%d.%s", imgs_directory, imgs_filename, k, extension);

				if(!DoesExist_(img_file))
					continue;

				Mat img = imread(img_file, IMREAD_COLOR);
				if(img.empty()) {
					cout << "Warning: Could not read image " << img_file << endl;
					continue;
				}
				
				if(img_size.empty())
					img_size = img.size();
					
				Mat gray;
				cv::cvtColor(img, gray, COLOR_BGR2GRAY);

				vector<Point2f> corners;
				bool found = findChessboardCorners(img, board_size, corners,
				                    CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);
				                    
				if (found) {
					cornerSubPix(gray, corners, Size(5, 5), Size(-1, -1),
					           TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
					drawChessboardCorners(gray, board_size, corners, found);
					
					vector<Point3f> obj;
					for (int i = 0; i < board_height; i++)
						for (int j = 0; j < board_width; j++)
							obj.push_back(Point3f((float)j * square_size, 
							                      (float)i * square_size, 0));

					cout << k << ". Found corners!" << endl;
					image_points_.push_back(corners);
					object_points_.push_back(obj);
					found_any = true;
				}
			}
			
			if(!found_any) {
				cout << "Error: No chessboard corners found in any image!" << endl;
				return false;
			}
			
			img_size_ = img_size;
			return true;
		}
		
		bool ComputeMatrixCalibration(int flag = CALIB_FIX_K4 | CALIB_FIX_K5) {
			if(object_points_.empty() || image_points_.empty()) {
				cout << "Error: No calibration data available. Run SetupCalibration first." << endl;
				return false;
			}
			
			if(img_size_.empty()) {
				cout << "Error: Image size not set." << endl;
				return false;
			}
			
			calibrateCamera(object_points_, image_points_, img_size_, K_, D_, rvecs_, tvecs_, flag);
			return true;
		}

		double ComputeReprojectionErrors() {
			if(object_points_.empty() || K_.empty()) {
				cout << "Error: No calibration data or camera matrix available." << endl;
				return -1.0;
			}
			
			vector<Point2f> imagePoints2;
			int totalPoints = 0;
			double totalErr = 0;
			
			for (size_t i = 0; i < object_points_.size(); ++i) {
				projectPoints(Mat(object_points_[i]), rvecs_[i], tvecs_[i], 
				              K_, D_, imagePoints2);
				double err = norm(Mat(image_points_[i]), Mat(imagePoints2), NORM_L2);
				int n = (int)object_points_[i].size();
				totalErr += err * err;
				totalPoints += n;
			}
			return std::sqrt(totalErr / totalPoints);
		}

		void SaveMatrixTo(const string& filename_out = "result_calibrating.yml") {
			if(K_.empty()) {
				cout << "Warning: Camera matrix is empty. Nothing to save." << endl;
				return;
			}
			
			FileStorage fs(filename_out, FileStorage::WRITE);
			if(!fs.isOpened()) {
				cout << "Error: Could not open file " << filename_out << " for writing." << endl;
				return;
			}
			
			fs << "K" << K_;
			fs << "D" << D_;
			fs << "board_width" << board_width_;
			fs << "board_height" << board_height_;
			fs << "square_size" << square_size_;
			cout << "Calibration saved to " << filename_out << endl;
		}
		
		Mat getCameraMatrix() const { return K_; }
		Mat getDistCoeffs() const { return D_; }
		vector<Mat> getRotationVectors() const { return rvecs_; }
		vector<Mat> getTranslationVectors() const { return tvecs_; }

	private:
		bool DoesExist_(const string& name) {
			struct stat buffer;   
			return (stat(name.c_str(), &buffer) == 0); 
		}

		vector<vector<Point3f>> object_points_;
		vector<vector<Point2f>> image_points_;
		vector<Mat> rvecs_, tvecs_;
		
		Mat K_, D_;
		Size img_size_;
		
		int board_width_, board_height_;
		float square_size_;
};