#include "stereo_pipeline.hpp"
#include "../.config/config.hpp"

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