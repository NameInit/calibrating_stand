#include "../include/camera_oak_manager.hpp"
#include <opencv2/opencv.hpp>

int main() {
    CameraOAKManager camera;
    
    if (!camera.IsOpened()) {
        std::cerr << "Failed to open camera" << std::endl;
        return 1;
    }
    
    while (true) {
        if (camera.Read()) {
            cv::Mat combined = camera.GetCombinedFrame();
            cv::imshow("OAK Stereo", combined);
            
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
    }
    
    cv::destroyAllWindows();
    return 0;
}