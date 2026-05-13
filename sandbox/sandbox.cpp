#include "params_manager.hpp"
#include "calib_intrinsic.hpp"
#include <iostream>
#include <string>
#include <vector>

void printYamlValue(const ParamsManager::YamlValue& val, int indent) {
    std::string spaces(indent, ' ');

    if (val.is<int>()) {
        std::cout << val.get<int>() << "\n";
    } 
    else if (val.is<double>()) {
        std::cout << val.get<double>() << "\n";
    } 
    else if (val.is<bool>()) {
        std::cout << (val.get<bool>() ? "true" : "false") << "\n";
    } 
    else if (val.is<std::string>()) {
        std::cout << "\"" << val.get<std::string>() << "\"\n";
    } 
    else if (val.is<ParamsManager::YamlArray>()) {
        const auto& arr = val.get<ParamsManager::YamlArray>();
        std::cout << "[";
        for (size_t i = 0; i < arr.size(); ++i) {
            printYamlValue(arr[i], 0);
            if (i < arr.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    } 
    else if (val.is<ParamsManager::YamlMap>()) {
        std::cout << "\n";
        const auto& sub_map = val.get<ParamsManager::YamlMap>();
        for (const auto& [key, sub_val] : sub_map) {
            std::cout << spaces << "  " << key << ": ";
            printYamlValue(sub_val, indent + 2);
        }
    }
}

int main() {
    auto &manager = ParamsManager::getInstance();
    if (!manager.load("../.config/params.yml")) {
        std::cerr << "Не удалось загрузить файл!" << std::endl;
        return -1;
    }

    std::cout << "=== ДАМП КОНФИГУРАЦИИ ИЗ MAIN ===\n";
    
    std::vector<std::string> root_keys = {
        "path", "camera", "roi", "vizualizer", "chessboard", "stereo_sgbm", "velocity_tracker"
    };

    for (const std::string& root_key : root_keys) {
        try {
            const auto& section = manager[root_key]; 
            std::cout << root_key << ": ";
            printYamlValue(section, 0);
        } catch (const std::exception& e) {
            std::cout << root_key << ": [Не найден в cache]\n";
        }
    }
    std::cout << "=================================\n";
    return 0;
}
