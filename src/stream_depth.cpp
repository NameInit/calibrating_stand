#include "stereo_pipeline.hpp"
#include "../.config/config.hpp"


//TO DO: подумать над конфигами
//TO DO: сделать конструктор со структурой
//TO DO: поиграться с параметрами
//TO DO: посмтреть сборку OpenCV с TBB, IPP, CUDA/cuDNN (если есть GPU)

// ORB-SLAM2 / ORB-SLAM3: Очень популярные, работают со стереокамерами. Дают точную оценку позы и строят карту. Сложно интегрировать, но дают выдающиеся результаты.
// RTAB-Map: Робастный SLAM, который может работать со стерео.
// OpenVSLAM: Еще одна SLAM-библиотека.
int main() {
    StereoPipeline pipeline(
        stereo_stream::calib_file,
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
        stereo_stream::median_blur_size,
        300, 300, 250, 250,
        stereo_stream::sgbm_num_disp
    );

    return pipeline.run();
}