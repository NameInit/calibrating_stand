#pragma once


#include <iostream>

#include "config_calibrating_cam.hpp"

namespace calibrating_stereo{
    std::string filename_left_cam_params = calibrating_cams::filename_left_cam_params;
    std::string filename_right_cam_params = calibrating_cams::filename_right_cam_params;
    constexpr int num_imgs = calibrating_cams::num_imgs;
    std::string img_left_filename = calibrating_cams::img_left_filename;
    std::string img_right_filename = calibrating_cams::img_right_filename;
    std::string path_imgs_directory = calibrating_cams::path_imgs_directory;
    std::string extension = calibrating_cams::extension;
    std::string out_file = "../result/cam_stereo.yml";
};