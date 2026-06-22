#pragma once
#include "params_manager.hpp"

struct StereoParams {
    int &min_disp;
    int &max_disp;
    int &block_size;
    int &uniqueness_ratio;
    int &speckle_ws;
    int &speckle_range;
    bool &use_wls;
    int &wls_lambda;
    double &wls_sigma;
    int &median_blur_size;

    StereoParams() :
        min_disp(ParamsManager::getInstance()["stereo_sgbm"]["min_disp"]),
        max_disp(ParamsManager::getInstance()["stereo_sgbm"]["max_disp"]),
        block_size(ParamsManager::getInstance()["stereo_sgbm"]["block_size"]),
        uniqueness_ratio(ParamsManager::getInstance()["stereo_sgbm"]["uniqueness_ratio"]),
        speckle_ws(ParamsManager::getInstance()["stereo_sgbm"]["speckle_ws"]),
        speckle_range(ParamsManager::getInstance()["stereo_sgbm"]["speckle_range"]),
        use_wls(ParamsManager::getInstance()["stereo_sgbm"]["use_wls"]),
        wls_lambda(ParamsManager::getInstance()["stereo_sgbm"]["wls_lambda"]),
        wls_sigma(ParamsManager::getInstance()["stereo_sgbm"]["wls_sigma"]),
        median_blur_size(ParamsManager::getInstance()["stereo_sgbm"]["median_blur_size"]) {}

    int getP1() const { return 8 * 3 * block_size * block_size; }
    int getP2() const { return 32 * 3 * block_size * block_size; }
};
