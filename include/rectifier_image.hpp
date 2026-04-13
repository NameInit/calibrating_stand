#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>


class RectifierImage{
	public:
		RectifierImage(const char* filename_calib_stereo)
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

		void UndistortRectify(const std::string& leftin_filename, const std::string& rightin_filename,
				const std::string& leftout_filename = "left_rect.png", const std::string& rightout_filename = "right_rect.png"){
			cv::Mat lmapx, lmapy, rmapx, rmapy;
			cv::Mat imgU1, imgU2;

			cv::Mat img1 = cv::imread(leftin_filename, cv::IMREAD_COLOR);
  			cv::Mat img2 = cv::imread(rightin_filename, cv::IMREAD_COLOR);

			initUndistortRectifyMap(K1_, D1_, R1_, P1_, img1.size(), CV_32F, lmapx, lmapy);
			initUndistortRectifyMap(K2_, D2_, R2_, P2_, img2.size(), CV_32F, rmapx, rmapy);
			remap(img1, imgU1, lmapx, lmapy, cv::INTER_LINEAR);
			remap(img2, imgU2, rmapx, rmapy, cv::INTER_LINEAR);

			imwrite(leftout_filename, imgU1);
			imwrite(rightout_filename, imgU2);

			return ;
		}
	private:
		cv::Mat R1_, R2_, P1_, P2_, Q_;
		cv::Mat K1_, K2_, R_;
		cv::Vec3d T_;
		cv::Mat D1_, D2_;
		cv::FileStorage fs1_;
};