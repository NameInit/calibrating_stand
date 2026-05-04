#include <iostream>
#include <chrono>
#include <string>
#include <iomanip>
#include <sstream>

class FPSCounter {
private:
    double update_interval;
    int counter_;
    double fps_;
    std::chrono::time_point<std::chrono::steady_clock> timer_;

public:
    FPSCounter(double update_interval_ = 1.0) 
        : update_interval(update_interval), counter_(0), fps_(0.0) {
        timer_ = std::chrono::steady_clock::now();
    }

    double update() {
        counter_++;
        auto current_time = std::chrono::steady_clock::now();
        
        std::chrono::duration<double> time_diff = current_time - timer_;

        if (time_diff.count() >= update_interval) {
            fps_ = counter_ / time_diff.count();
            counter_ = 0;
            timer_ = current_time;
        }

        return fps_;
    }

    double get_fps(){
        return fps_;
    } 

    std::string get_fps_text() const {
        std::stringstream ss;
        ss << "FPS: " << std::fixed << std::setprecision(1) << fps_;
        return ss.str();
    }
};
