#include "../include/calib_intrinsic.hpp"
#include "../include/calib_stereo.hpp"
#include "../include/rectifier_image.hpp"
#include "../.config/config.hpp"


int main(){
    CalibratingCam calibrating_left_cam, calibrating_right_cam;
    //TO DO: разбить на 2 потока
    std::cout << "---------START CALIBRATE LEFT CAM---------" << std::endl;
    calibrating_left_cam.SetupCalibration(
        calibrating_cams::board_width,
        calibrating_cams::board_height,
        calibrating_cams::num_imgs,
        calibrating_cams::square_size,
        calibrating_cams::path_imgs_directory.data(),
        calibrating_cams::img_left_filename.data(),
        calibrating_cams::extension.data()
    );
    calibrating_left_cam.ComputeMatrixCalibration();
    std::cout << "Error pixel = " << calibrating_left_cam.ComputeReprojectionErrors() << std::endl;
    calibrating_left_cam.SaveMatrixTo(calibrating_cams::filename_left_cam_params);
    std::cout << "----------END CALIBRATE LEFT CAM----------" << std::endl;

    std::cout << "---------START CALIBRATE RIGHT CAM---------" << std::endl;
    calibrating_right_cam.SetupCalibration(
        calibrating_cams::board_width,
        calibrating_cams::board_height,
        calibrating_cams::num_imgs,
        calibrating_cams::square_size,
        calibrating_cams::path_imgs_directory.data(),
        calibrating_cams::img_right_filename.data(),
        calibrating_cams::extension.data()
    );
    calibrating_right_cam.ComputeMatrixCalibration();
    std::cout << "Error pixel = " << calibrating_right_cam.ComputeReprojectionErrors() << std::endl;
    calibrating_right_cam.SaveMatrixTo(calibrating_cams::filename_right_cam_params);
    std::cout << "----------END CALIBRATE RIGHT CAM----------" << std::endl;
    //

    std::cout << "---------START CALIBRATE STEREO CAM---------" << std::endl;
    CalibratingStereo calibrating_stereo(calibrating_stereo::filename_left_cam_params.data(),calibrating_stereo::filename_right_cam_params.data());
    calibrating_stereo.LoadImagePoints(
        calibrating_stereo::num_imgs,
        calibrating_stereo::path_imgs_directory.data(),
        calibrating_stereo::path_imgs_directory.data(),
        calibrating_stereo::img_left_filename.data(),
        calibrating_stereo::img_right_filename.data(),
        calibrating_stereo::extension.data()
    );

    //TODO: разбить на 2 потока
    calibrating_stereo.ComputeMatrixStereo();
    calibrating_stereo.ComputeMatrixStereoRectify();
    //
    calibrating_stereo.SaveMatrixTo(calibrating_stereo::out_file);
    std::cout << "---------END CALIBRATE STEREO CAM---------" << std::endl;

    RectifierImage rect(calibrating_stereo::out_file.data());
    rect.UndistortRectify(
        rectifier::filename_imleft_in,
        rectifier::filename_imright_in,
        rectifier::filename_imleft_out,
        rectifier::filename_imright_out
    );
    return 0;
}