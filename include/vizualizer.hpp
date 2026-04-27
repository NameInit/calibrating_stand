#pragma once


#include <opencv2/opencv.hpp>
#include <iostream>

class Vizualizer{
    private:
        std::string win_name_;
        double max_disp_;

        cv::Mat frame_vis_;
        cv::Size size_frame_vis_;
    public:
        Vizualizer(const std::string& win_name, double max_disp, const cv::Size& size_frame_vis = {1280,800})
            : win_name_(win_name), max_disp_(max_disp), size_frame_vis_(size_frame_vis){
            cv::namedWindow(win_name_, cv::WINDOW_AUTOSIZE);
        }
        ~Vizualizer(){
            cv::destroyWindow(win_name_);
        }

        Vizualizer& render(const cv::Mat& cam_frame, const cv::Mat& disp_map, 
            const cv::Rect& roi_rect, double fps, double dist, double vel_kmh){
            
            //меняем число каналов на 3
            if(cam_frame.channels()==1){
                cv::cvtColor(cam_frame, frame_vis_, cv::COLOR_GRAY2BGR);
            }
            else if(cam_frame.channels()==3){
                frame_vis_=cam_frame.clone();
            }
            else if(cam_frame.channels()==4){
                cv::cvtColor(cam_frame, frame_vis_, cv::COLOR_RGBA2BGR);
            }
            else{
                std::cerr << "Error count channel in cam_frame" << std::endl;
            }

            cv::Mat disp_vis_roi, roi = disp_map(roi_rect);
            roi.convertTo(disp_vis_roi, CV_8U, 255/max_disp_);
            cv::applyColorMap(disp_vis_roi,disp_vis_roi,cv::COLORMAP_JET);

            disp_vis_roi.copyTo(frame_vis_(roi_rect));
            cv::rectangle(frame_vis_, roi_rect, cv::Scalar(255, 255, 255), 1);

            if(frame_vis_.size()!=size_frame_vis_){
                cv::resize(frame_vis_,frame_vis_,size_frame_vis_);
            }

            std::string label = cv::format("FPS: %.1lf | DIST_CENTER: %.2lf m | VEL: %.1lf km/h", 
                                        fps, dist, vel_kmh);
            cv::putText(frame_vis_, label, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 3);
            return *this;
        }

        Vizualizer& show(){
            cv::imshow(win_name_, frame_vis_);
            return *this;
        }
};