#include <iostream>
#include <opencv2/opencv.hpp>

#include "calib_intrinsic.hpp"
#include "calib_stereo.hpp"
#include "rectifier_image.hpp"
#include "params_manager.hpp"

int main(){
    ParamsManager::getInstance().load();

    std::cout << "---------START CALIBRATE LEFT CAM---------" << std::endl;
    DatasetParams p_left;
    p_left.imgs_directory=ParamsManager::getInstance()["path"]["left_imgs_directory"].get<std::string>();
    p_left.imgs_filename=ParamsManager::getInstance()["path"]["img_left_filename"].get<std::string>();
    CalibratingCam calibrating_left_cam;
    calibrating_left_cam.SetupCalibration(p_left);
    calibrating_left_cam.ComputeMatrixCalibration();
    std::cout << "Error pixel = " << calibrating_left_cam.ComputeReprojectionErrors() << std::endl;
    calibrating_left_cam.SaveMatrixTo(ParamsManager::getInstance()["path"]["out_params"].get<std::string>()
                    +ParamsManager::getInstance()["path"]["filename_left_cam_params"].get<std::string>());
    std::cout << "----------END CALIBRATE LEFT CAM----------" << std::endl;

    std::cout << "---------START CALIBRATE RIGHT CAM---------" << std::endl;
    DatasetParams p_right;
    p_right.imgs_directory=ParamsManager::getInstance()["path"]["right_imgs_directory"].get<std::string>();
    p_right.imgs_filename=ParamsManager::getInstance()["path"]["img_right_filename"].get<std::string>();
    CalibratingCam calibrating_right_cam;
    calibrating_right_cam.SetupCalibration(p_right);
    calibrating_right_cam.ComputeMatrixCalibration();
    std::cout << "Error pixel = " << calibrating_right_cam.ComputeReprojectionErrors() << std::endl;
    calibrating_right_cam.SaveMatrixTo(ParamsManager::getInstance()["path"]["out_params"].get<std::string>()
                    +ParamsManager::getInstance()["path"]["filename_right_cam_params"].get<std::string>());
    std::cout << "----------END CALIBRATE RIGHT CAM----------" << std::endl;

    ParamsManager::getInstance().refresh(); //для сброса cсылок на базовые

    std::cout << "---------START CALIBRATE STEREO CAM---------" << std::endl;
    CalibratingStereo calibrating_stereo;
    calibrating_stereo.LoadImagePoints();

    calibrating_stereo.ComputeMatrixStereo();
    calibrating_stereo.ComputeMatrixStereoRectify();

    calibrating_stereo.SaveMatrixTo(ParamsManager::getInstance()["path"]["out_params"].get<std::string>()
                    +ParamsManager::getInstance()["path"]["filename_stereo_cam_params"].get<std::string>());
    std::cout << "---------END CALIBRATE STEREO CAM---------" << std::endl;

    RectifierImage rect(ParamsManager::getInstance()["path"]["out_params"].get<std::string>()
                    +ParamsManager::getInstance()["path"]["filename_stereo_cam_params"].get<std::string>());
    rect.UndistortRectify();
    return 0;
}