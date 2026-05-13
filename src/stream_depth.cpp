#include "stereo_pipeline.hpp"
#include "params_manager.hpp"

int main() {
    ParamsManager::getInstance().load("../.config/params.yml");

    StereoPipeline pipeline("../result/cam_stereo.yml");

    return pipeline.run();
}