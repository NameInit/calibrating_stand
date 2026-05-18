#include "stereo_pipeline.hpp"
#include "params_manager.hpp"

int main() {
    ParamsManager::getInstance().load();

    StereoPipeline pipeline(
        ParamsManager::getInstance()["path"]["out_params"].get<std::string>()
        +ParamsManager::getInstance()["path"]["filename_stereo_cam_params"].get<std::string>()
    );

    return pipeline.run();
}