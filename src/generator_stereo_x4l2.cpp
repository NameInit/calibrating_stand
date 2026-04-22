#include <iostream>
#include <opencv2/opencv.hpp>
#include "../.config/config.hpp"

int main(){
    std::string out_dir = "../data/image/chessboard_10_7_paper_st_1/";
    cv::VideoCapture cap(2, cv::CAP_V4L2);

    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open cameras" << std::endl;
        return 1;
    }

    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G')); //buffer
    cap.set(cv::CAP_PROP_FRAME_WIDTH, stereo_stream::img_width*2);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, stereo_stream::img_height);

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
            cv::imwrite(out_dir+"left"+std::to_string(count)+".png", left_img);
            cv::imwrite(out_dir+"right"+std::to_string(count)+".png", right_img);
            // std::cout << out_dir+"left"+std::to_string(count)+".png" << std::endl;
        }
        
    }
    cv::destroyAllWindows();
    std::cout << "Saved " << count << " picture" << std::endl;
    return 0;
}