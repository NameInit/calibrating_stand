#include <iostream>
#include <opencv2/opencv.hpp>

#include "calib_intrinsic.hpp"
#include "calib_stereo.hpp"
#include "rectifier_image.hpp"

int main(){
    cv::FileStorage fs("../.config/params.yml", cv::FileStorage::READ);
    cv::FileNode path_node = fs["path"];

    std::cout << "---------START CALIBRATE LEFT CAM---------" << std::endl;
    DatasetParams p_left;
    p_left.imgs_filename=static_cast<std::string>(path_node["img_left_filename"]);
    CalibratingCam calibrating_left_cam;
    calibrating_left_cam.SetupCalibration(p_left);
    calibrating_left_cam.ComputeMatrixCalibration();
    std::cout << "Error pixel = " << calibrating_left_cam.ComputeReprojectionErrors() << std::endl;
    calibrating_left_cam.SaveMatrixTo("../result/cam_left.yml");
    std::cout << "----------END CALIBRATE LEFT CAM----------" << std::endl;

    std::cout << "---------START CALIBRATE RIGHT CAM---------" << std::endl;
    DatasetParams p_right;
    p_left.imgs_filename=static_cast<std::string>(path_node["img_right_filename"]);
    CalibratingCam calibrating_right_cam;
    calibrating_right_cam.SetupCalibration(p_right);
    calibrating_right_cam.ComputeMatrixCalibration();
    std::cout << "Error pixel = " << calibrating_right_cam.ComputeReprojectionErrors() << std::endl;
    calibrating_right_cam.SaveMatrixTo("../result/cam_right.yml");
    std::cout << "----------END CALIBRATE RIGHT CAM----------" << std::endl;

    std::cout << "---------START CALIBRATE STEREO CAM---------" << std::endl;
    CalibratingStereo calibrating_stereo;
    calibrating_stereo.LoadImagePoints();

    calibrating_stereo.ComputeMatrixStereo();
    calibrating_stereo.ComputeMatrixStereoRectify();

    calibrating_stereo.SaveMatrixTo("../result/cam_stereo.yml");
    std::cout << "---------END CALIBRATE STEREO CAM---------" << std::endl;

    RectifierImage rect("../result/cam_stereo.yml");
    rect.UndistortRectify();
    return 0;
}