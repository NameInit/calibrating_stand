#pragma once


#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>

namespace stereo_stream{
    std::string calib_file = "../result/cam_stereo.yml";
    int img_width = 1280;
    int img_height = 800;
    int fps = 30;
    bool use_wls_filter = true; 
    int wls_lambda = 80000;
    float wls_sigma = 1.5;
    int sgbm_num_disp = 80;
    int sgbm_block_size = 5;
    cv::Size img_size(img_width, img_height);
    int min_disp = 0;
    int p1=8 * 3 * sgbm_block_size*sgbm_block_size;
    int p2=32 * 3 * sgbm_block_size*sgbm_block_size;
    int sgbm_uniqueness_ratio=10;
    int sgbm_speckle_ws=100;
    int sgbm_speckle_range = 32;
    int median_blur_size=0;
};