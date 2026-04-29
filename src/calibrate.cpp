#include "../include/calib_intrinsic.hpp"
#include "../include/calib_stereo.hpp"
#include "../include/rectifier_image.hpp"
#include "../.config/config.hpp"


int main(){
    std::cout << "---------START CALIBRATE LEFT CAM---------" << std::endl;
    DatasetParams p_left;
    p_left.imgs_filename="left";
    CalibratingCam calibrating_left_cam;
    calibrating_left_cam.SetupCalibration(p_left);
    calibrating_left_cam.ComputeMatrixCalibration();
    std::cout << "Error pixel = " << calibrating_left_cam.ComputeReprojectionErrors() << std::endl;
    calibrating_left_cam.SaveMatrixTo(calibrating_cams::filename_left_cam_params);
    std::cout << "----------END CALIBRATE LEFT CAM----------" << std::endl;

    std::cout << "---------START CALIBRATE RIGHT CAM---------" << std::endl;
    DatasetParams p_right;
    p_left.imgs_filename="right";
    CalibratingCam calibrating_right_cam;
    calibrating_right_cam.SetupCalibration(p_right);
    calibrating_right_cam.ComputeMatrixCalibration();
    std::cout << "Error pixel = " << calibrating_right_cam.ComputeReprojectionErrors() << std::endl;
    calibrating_right_cam.SaveMatrixTo(calibrating_cams::filename_right_cam_params);
    std::cout << "----------END CALIBRATE RIGHT CAM----------" << std::endl;

    std::cout << "---------START CALIBRATE STEREO CAM---------" << std::endl;
    CalibratingStereo calibrating_stereo;
    calibrating_stereo.LoadImagePoints();

    calibrating_stereo.ComputeMatrixStereo();
    calibrating_stereo.ComputeMatrixStereoRectify();

    calibrating_stereo.SaveMatrixTo(calibrating_stereo::out_file);
    std::cout << "---------END CALIBRATE STEREO CAM---------" << std::endl;

    RectifierImage rect(calibrating_stereo::out_file);
    rect.UndistortRectify();
    return 0;
}