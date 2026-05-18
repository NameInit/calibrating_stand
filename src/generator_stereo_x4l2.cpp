#include <iostream>
#include <opencv2/opencv.hpp>

#include "params_manager.hpp"

int main(){
    ParamsManager::getInstance().load();

    cv::VideoCapture cap(
        ParamsManager::getInstance()["camera"]["video_id"].get<int>(),
        ParamsManager::getInstance()["camera"]["api_preference"].get<int>()
    );

    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open cameras" << std::endl;
        return 1;
    }


    if(ParamsManager::getInstance()["camera"]["use_mjpg"].get<bool>()){
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G')); //buffer
    }

    cv::Size size_frame = ParamsManager::getInstance()["camera"]["size_im"].get<cv::Size>();
    cap.set(cv::CAP_PROP_FRAME_WIDTH, size_frame.width*2);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, size_frame.height);

    cv::Mat frame, left_img, right_img;
    int count=0;
    while (cap.read(frame)) {
        int width = frame.cols / 2;
        left_img = frame(cv::Rect(0, 0, width, frame.rows));
        right_img = frame(cv::Rect(width, 0, width, frame.rows));

        // cv::imshow("Left", left_img);
        // cv::imshow("Right", right_img);

        cv::Mat gray_left_img, gray_right_img;
        cv::cvtColor(left_img, gray_left_img, cv::COLOR_RGB2GRAY);
        cv::cvtColor(right_img, gray_right_img, cv::COLOR_RGB2GRAY);

        // cv::imshow("GrayLeft", gray_left_img);
        // cv::imshow("GrayRight", gray_right_img);
        cv::Mat vis;
        cv::hconcat(gray_left_img,gray_right_img,vis);
        cv::resize(vis,vis,{800,400});
        cv::imshow("ConcatGray", vis);

        int key = cv::waitKey(1);
        if (key == 'q') break;
        else if (key == 32) {
            ++count;
            std::cout << "save" << count << std::endl;
            cv::imwrite(ParamsManager::getInstance()["path"]["left_imgs_directory"].get<std::string>()+"left"+std::to_string(count)+".png", left_img);
            cv::imwrite(ParamsManager::getInstance()["path"]["right_imgs_directory"].get<std::string>()+"right"+std::to_string(count)+".png", right_img);
        }
        
    }
    cv::destroyAllWindows();
    std::cout << "Saved " << count << " picture" << std::endl;
    return 0;
}