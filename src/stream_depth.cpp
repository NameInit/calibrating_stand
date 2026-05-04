#include "stereo_pipeline.hpp"


int main() {
    StereoPipeline pipeline(
        "../result/cam_stereo.yml"
    );

    return pipeline.run();
}