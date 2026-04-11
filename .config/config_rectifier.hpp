#pragma once


#include <iostream>
#include "config_calibrating_cam.hpp"


namespace rectifier{
    std::string path_out = "../result/";
    std::string filename_imleft_in = calibrating_cams::path_imgs_directory+"left1"+"."+calibrating_cams::extension;
    std::string filename_imright_in = calibrating_cams::path_imgs_directory+"right1"+"."+calibrating_cams::extension;
    std::string filename_imleft_out = path_out+"left_rect"+"."+calibrating_cams::extension;
    std::string filename_imright_out = path_out+"right_rect"+"."+calibrating_cams::extension;
};