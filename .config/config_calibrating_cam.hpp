#pragma once


#include <iostream>

#include "config_chessboard.hpp"

namespace calibrating_cams{    
    constexpr int board_width = chessboard::board_width;
    constexpr int board_height = chessboard::board_height;
    constexpr float square_size = chessboard::square_size;
    constexpr int num_imgs = 72;

    std::string path_imgs_directory = "../data/image/chessboard_10_7_paper_st_1280_800/";
    std::string img_left_filename = "left";
    std::string img_right_filename = "right";
    std::string extension = "png";
    std::string filename_left_cam_params = "../result/cam_left.yml";
    std::string filename_right_cam_params = "../result/cam_right.yml";
};