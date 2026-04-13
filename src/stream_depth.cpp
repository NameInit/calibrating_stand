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

    if (!camera.IsOpened()) {
        std::cerr << "Failed to open camera" << std::endl;
        return 1;
    }

    int cur_x=300, cur_y=300;
    int win_width = 250, win_height = 250;

    while (true) {
        if (camera.Read()) {
            cv::Mat left_raw = camera.GetLeftFrame();
            cv::Mat right_raw = camera.GetRightFrame();
            
            if (left_raw.empty() || right_raw.empty()) continue;

            if(left_raw.channels()==3){
                cv::cvtColor(left_raw, left_raw, cv::COLOR_BGR2GRAY);
            }
            if(right_raw.channels() == 3){
                cv::cvtColor(right_raw, right_raw, cv::COLOR_BGR2GRAY);
            }

            cv::Mat left_rect, right_rect;
            stereoSGBM.Rectify(left_raw, right_raw, left_rect, right_rect);

            
            cv::Rect roi(cur_x - win_width/2, cur_y - win_height/2, win_width, win_height);
            roi &= cv::Rect(0, 0, left_rect.cols, left_rect.rows);

            cv::Mat left_roi = left_rect(roi);
            cv::Mat right_roi = right_rect(roi);

            cv::Mat disp_roi = stereoSGBM.Compute(left_roi, right_roi);

            cv::Mat disp_vis_roi;
            
            cv::cvtColor(left_rect,left_rect,cv::COLOR_GRAY2RGB);

            if (!disp_roi.empty()) {
                double alpha = 255.0 / stereo_stream::sgbm_num_disp;
                disp_roi.convertTo(disp_vis_roi, CV_8U, alpha);
                cv::normalize(disp_vis_roi, disp_vis_roi, 0, 255, cv::NORM_MINMAX, CV_8U);
                cv::applyColorMap(disp_vis_roi, disp_vis_roi, cv::COLORMAP_JET);
                disp_vis_roi.copyTo(left_rect(roi));

                cv::Mat depth_roi = stereoSGBM.GetDepthMap(disp_roi);
                float dist = depth_roi.at<float>(depth_roi.rows / 2, depth_roi.cols / 2);

                std::string label = cv::format("To center: %.2f m", dist);
                cv::putText(left_rect, label, cv::Point(roi.x, roi.y - 5), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            }
            
            cv::rectangle(left_rect, roi, cv::Scalar(255, 255, 255), 1);

            cv::imshow("Left Rectified", left_rect);

            int key = cv::waitKey(1);
            if (key == 'q') { break; }
            else if (key == 'a' || key == 'A') { cur_x -= 30; }
            else if (key == 'd' || key == 'D') { cur_x += 30; }
            else if (key == 'w' || key == 'W') { cur_y -= 30; }
            else if (key == 's' || key == 'S') { cur_y += 30; }
            else if (key == 'c' || key == 'C') { win_width -= 30; }
            else if (key == 'v' || key == 'V') { win_width += 30; }
            else if (key == 'z' || key == 'Z') { win_height -= 30; }
            else if (key == 'x' || key == 'X') { win_height += 30; }

            if (cur_x < 0) cur_x = 0;
            if (cur_x > left_rect.cols) cur_x = left_rect.cols;
            if (cur_y < 0) cur_y = 0;
            if (cur_y > left_rect.rows) cur_y = left_rect.rows;
            if (win_width < 100) win_width = 100;
            if (win_width > left_rect.cols) win_width = left_rect.cols;
            if (win_height < 100) win_height = 100;
            if (win_height > left_rect.rows) win_height = left_rect.rows;
        }
    }
    
    cv::destroyAllWindows();
    return 0;
}