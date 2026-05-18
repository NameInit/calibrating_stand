#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>


struct VizualizerParams{
    std::string win_name;
    double max_disp;
    cv::Size size_frame_vis;

    VizualizerParams(const std::string& config_name = "../.config/params.yml"){
        cv::FileStorage fs(config_name, cv::FileStorage::READ);
		cv::FileNode vizualizer_node = fs["vizualizer"];
        cv::FileNode stereo_sgbm_node = fs["stereo_sgbm"];

        vizualizer_node["win_name"] >> win_name;
        stereo_sgbm_node["max_disp"] >> max_disp;
        vizualizer_node["size_frame_vis"] >> size_frame_vis;
    }
};

class Vizualizer{
    private:
        cv::Mat frame_vis_;
        VizualizerParams params_;
    public:
        Vizualizer(){
            cv::namedWindow(params_.win_name, cv::WINDOW_AUTOSIZE);
        }
        ~Vizualizer(){
            cv::destroyWindow(params_.win_name);
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
            roi.convertTo(disp_vis_roi, CV_8U, 255/params_.max_disp);
            cv::applyColorMap(disp_vis_roi,disp_vis_roi,cv::COLORMAP_JET);

            disp_vis_roi.copyTo(frame_vis_(roi_rect));
            cv::rectangle(frame_vis_, roi_rect, cv::Scalar(255, 255, 255), 1);

            if(frame_vis_.size()!=params_.size_frame_vis){
                cv::resize(frame_vis_,frame_vis_,params_.size_frame_vis);
            }

            std::string label = cv::format("FPS: %.1lf | DIST_CENTER: %.2lf m | VEL: %.1lf km/h", 
                                        fps, dist, vel_kmh);
            cv::putText(frame_vis_, label, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 3);
            return *this;
        }

        Vizualizer& show(){
            if(!frame_vis_.empty()){
                cv::imshow(params_.win_name, frame_vis_);
            }
            return *this;
        }

        VizualizerParams& getParams(){
            return params_;
        }
};