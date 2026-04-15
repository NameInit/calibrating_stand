#include <iostream>
#include <chrono>
#include <string>
#include <iomanip>
#include <sstream>

class FPSCounter {
private:
    double update_interval;
    int counter;
    double fps;
    std::chrono::time_point<std::chrono::steady_clock> timer;

public:
    FPSCounter(double update_interval = 1.0) 
        : update_interval(update_interval), counter(0), fps(0.0) {
        timer = std::chrono::steady_clock::now();
    }

    double update() {
        counter++;
        auto current_time = std::chrono::steady_clock::now();
        
        std::chrono::duration<double> time_diff = current_time - timer;

        if (time_diff.count() >= update_interval) {
            fps = counter / time_diff.count();
            counter = 0;
            timer = current_time;
        }

        return fps;
    }

    double get_fps(){
        return fps;
    } 

    std::string get_fps_text() const {
        std::stringstream ss;
        ss << "FPS: " << std::fixed << std::setprecision(1) << fps;
        return ss.str();
    }
};
