#include <opencv2/opencv.hpp>
#include <iostream>

#include "camera_oak_manager.hpp"
#include "stereo_sgbm.hpp"
#include "../.config/config.hpp"

int main() {
    CameraOAKManager camera;
    StereoSGBM stereoSGBM(stereo_stream::calib_file);

    stereoSGBM.Create(
        stereo_stream::min_disp,
        stereo_stream::sgbm_num_disp,
        stereo_stream::sgbm_block_size,
        stereo_stream::p1,
        stereo_stream::p2,
        stereo_stream::sgbm_uniqueness_ratio,
        stereo_stream::sgbm_speckle_ws,
        stereo_stream::sgbm_speckle_range,
        stereo_stream::use_wls_filter,
        stereo_stream::wls_lambda,
        stereo_stream::wls_sigma,
        stereo_stream::median_blur_size
    );

    stereoSGBM.InitRectification(stereo_stream::img_size);

    if (!camera.IsOpened()) {
        std::cerr << "Failed to open camera" << std::endl;
        return 1;
    }

    
    
    while (true) {
        if (camera.Read()) {
            cv::Mat left_frame = camera.GetLeftFrame();
            cv::Mat right_frame = camera.GetRightFrame();
            cv::Mat combined_frame = camera.GetCombinedFrame();

            // int win_size = 200;
            // cv::Mat black_screen = cv::Mat::zeros(left_frame.size(), CV_8UC3);
            // cv::Rect roi(300 - win_size/2, 300 - win_size/2, win_size, win_size);
            // roi &= cv::Rect(0, 0, left_frame.cols, left_frame.rows);
            // cv::Mat left_roi = left_frame(roi);
            // cv::Mat right_roi = right_frame(roi);
            // cv::Mat disp_roi = stereoSGBM.Compute(left_roi, right_roi);
            // cv::Mat disp_vis_roi;
            // cv::normalize(disp_roi, disp_vis_roi, 0, 255, cv::NORM_MINMAX, CV_8U);
            // cv::applyColorMap(disp_vis_roi, disp_vis_roi, cv::COLORMAP_JET);
            // disp_vis_roi.copyTo(black_screen(roi));
            // cv::rectangle(black_screen, roi, cv::Scalar(255, 255, 255), 1);
            // cv::imshow("Focused Stereo", black_screen);

            // if (left_frame.empty() || right_frame.empty()) continue;

            cv::Mat disp_vis, 
                    disparity = stereoSGBM.Compute(left_frame, right_frame),
                    depth = stereoSGBM.GetDepthMap(disparity);

            cv::normalize(disparity, disp_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::applyColorMap(disp_vis, disp_vis, cv::COLORMAP_JET);
            
            cv::imshow("Left", left_frame);
            cv::imshow("OAK Stereo", combined_frame);
            
            float dist = depth.at<float>(depth.rows / 2, depth.cols / 2);
            std::cout << "Distance to center: " << dist << " meters" << std::endl;

            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
    }
    
    cv::destroyAllWindows();
    return 0;
}