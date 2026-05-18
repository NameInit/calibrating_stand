#pragma once

#include <iostream>
#include <map>
#include <variant>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <opencv2/opencv.hpp>

class ParamsManager {
public:
    struct YamlValue;
    using YamlMap = std::map<std::string, YamlValue>;
    using YamlArray = std::vector<YamlValue>;

    struct YamlValue {
        std::variant<int, double, bool, std::string, cv::Size, YamlArray, YamlMap> data;

        template<typename T> bool is() const { return std::holds_alternative<T>(data); }
        template<typename T> const T& get() const { return std::get<T>(data); }

        YamlValue& operator[](const std::string& key) {
            if (!std::holds_alternative<YamlMap>(data)) {
                throw std::runtime_error("[ParamsManager] Узел не является секцией (YamlMap)");
            }
            return std::get<YamlMap>(data)[key];
        }

        YamlValue& operator[](const char* key) {
            return operator[](std::string(key));
        }

        YamlValue& operator[](size_t index) {
            if (!std::holds_alternative<YamlArray>(data)) {
                throw std::runtime_error("[ParamsManager] Узел не является массивом (YamlArray)");
            }
            return std::get<YamlArray>(data)[index];
        }

        const YamlValue& operator[](const std::string& key) const {
            if (!std::holds_alternative<YamlMap>(data)) {
                throw std::runtime_error("[ParamsManager] Узел не является секцией (YamlMap)");
            }
            return std::get<YamlMap>(data).at(key);
        }

        const YamlValue& operator[](const char* key) const {
            return operator[](std::string(key));
        }

        const YamlValue& operator[](size_t index) const {
            if (!std::holds_alternative<YamlArray>(data)) {
                throw std::runtime_error("[ParamsManager] Узел не является массивом (YamlArray)");
            }
            return std::get<YamlArray>(data).at(index);
        }

        template<typename T>
        operator T&() {
            try {
                return std::get<T>(data);
            } catch (const std::bad_variant_access&) {
                throw std::runtime_error("[ParamsManager] Ошибка приведения типов в std::variant");
            }
        }

        template<typename T>
        operator const T&() const {
            try {
                return std::get<T>(data);
            } catch (const std::bad_variant_access&) {
                throw std::runtime_error("[ParamsManager] Ошибка приведения константных типов в std::variant");
            }
        }
    };

    static ParamsManager& getInstance() {
        static ParamsManager instance;
        return instance;
    }

    ParamsManager(const ParamsManager&) = delete;
    ParamsManager& operator=(const ParamsManager&) = delete;

private:
    ParamsManager() = default;

    YamlMap cache;
    std::string last_filepath;

    void updateNode_(YamlValue& existing_node, const cv::FileNode& new_node) {
        if (new_node.isMap()) {
            if (!std::holds_alternative<YamlMap>(existing_node.data)) {
                existing_node = parseNode_(new_node);
                return;
            }
            YamlMap& existing_map = std::get<YamlMap>(existing_node.data);
            for (auto it = new_node.begin(); it != new_node.end(); ++it) {
                cv::FileNode sub_node = *it;
                std::string key = sub_node.name();
                if (existing_map.find(key) != existing_map.end()) {
                    updateNode_(existing_map[key], sub_node);
                } else {
                    existing_map[key] = parseNode_(sub_node);
                }
            }
        } 
        else if (new_node.isSeq()) {
            existing_node = parseNode_(new_node);
        } 
        else {
            if (new_node.isInt()) {
                if (!std::holds_alternative<int>(existing_node.data)) {
                    std::cerr << "[ParamsManager] Предупреждение: Ожидался тип int, тип в файле изменен!" << std::endl;
                }
                existing_node.data = static_cast<int>(new_node);
            } 
            else if (new_node.isReal()) {
                if (!std::holds_alternative<double>(existing_node.data)) {
                    std::cerr << "[ParamsManager] Предупреждение: Ожидался тип double, тип в файле изменен!" << std::endl;
                }
                existing_node.data = static_cast<double>(new_node);
            } 
            else if (new_node.isString()) {
                std::string str = static_cast<std::string>(new_node);
                if (str == "true" || str == "True" || str == "false" || str == "False") {
                    if (!std::holds_alternative<bool>(existing_node.data)) {
                        std::cerr << "[ParamsManager] Предупреждение: Ожидался тип bool, тип в файле изменен!" << std::endl;
                    }
                    existing_node.data = (str == "true" || str == "True");
                } else {
                    if (!std::holds_alternative<std::string>(existing_node.data)) {
                        std::cerr << "[ParamsManager] Предупреждение: Ожидался тип string, тип в файле изменен!" << std::endl;
                    }
                    existing_node.data = str;
                }
            }
            else if (new_node.isSeq() && std::holds_alternative<cv::Size>(existing_node.data)) {
                YamlValue fresh = parseNode_(new_node);
                if (std::holds_alternative<cv::Size>(fresh.data)) {
                    std::get<cv::Size>(existing_node.data) = std::get<cv::Size>(fresh.data);
                }
            }
        }
    }

    YamlValue parseNode_(const cv::FileNode& node) {
        YamlValue result;

        if (node.isMap()) {
            YamlMap nested_map;
            for (auto it = node.begin(); it != node.end(); ++it) {
                cv::FileNode sub_node = *it;
                nested_map[sub_node.name()] = parseNode_(sub_node);
            }
            result.data = nested_map;
        }
        else if (node.isSeq()) {
            YamlArray arr;
            for (auto it = node.begin(); it != node.end(); ++it) {
                arr.push_back(parseNode_(*it));
            }
            if (arr.size() == 2 && arr[0].is<int>() && arr[1].is<int>()) {
                result.data = cv::Size(arr[0].get<int>(), arr[1].get<int>());
            } else {
                result.data = arr;
            }
        } 
        else if (node.isInt()) {
            result.data = static_cast<int>(node);
        } 
        else if (node.isReal()) {
            result.data = static_cast<double>(node);
        } 
        else if (node.isString()) {
            std::string str = static_cast<std::string>(node);
            if (str == "true" || str == "True") result.data = true;
            else if (str == "false" || str == "False") result.data = false;
            else result.data = str;
        }

        return result;
    }

public:
    YamlValue& operator[](const std::string& key) {
        auto it = cache.find(key);
        if (it == cache.end()) {
            throw std::runtime_error("[ParamsManager] Корневой ключ '" + key + "' не найден в cache");
        }
        return it->second;
    }

    YamlValue& operator[](const char* key) {
        return operator[](std::string(key));
    }

    const YamlValue& operator[](const std::string& key) const {
        auto it = cache.find(key);
        if (it == cache.end()) {
            throw std::runtime_error("[ParamsManager] Корневой ключ '" + key + "' не найден в cache");
        }
        return it->second;
    }

    const YamlValue& operator[](const char* key) const {
        return operator[](std::string(key));
    }

    bool refresh() {
        if (last_filepath.empty()) {
            std::cerr << "[ParamsManager] refresh() вызван до load()" << std::endl;
            return false;
        }

        cv::FileStorage fs(last_filepath, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "[ParamsManager] Ошибка открытия файла при обновлении: " << last_filepath << std::endl;
            return false;
        }

        cv::FileNode root = fs.root();
        for (auto it = root.begin(); it != root.end(); ++it) {
            cv::FileNode section = *it;
            std::string section_name = section.name();
            if (cache.find(section_name) != cache.end()) {
                updateNode_(cache[section_name], section);
            } else {
                cache[section_name] = parseNode_(section);
            }
        }
        return true;
    }

    bool load(const std::string& filepath = "../.config/params.yml") {
        cv::FileStorage fs(filepath, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "[ParamsManager] Ошибка открытия файла: " << filepath << std::endl;
            return false;
        }

        last_filepath = filepath;
        cache.clear();
        cv::FileNode root = fs.root();
        for (auto it = root.begin(); it != root.end(); ++it) {
            cv::FileNode section = *it;
            cache[section.name()] = parseNode_(section);
        }
        return true;
    }

    void clear() {
        cache.clear();
        last_filepath.clear();
    }

    template<typename T>
    T get(const std::string& path) const {
        std::stringstream ss(path);
        std::string item;
        const YamlMap* current_map = &cache;
        
        while (std::getline(ss, item, '.')) {
            auto it = current_map->find(item);
            if (it == current_map->end()) {
                throw std::runtime_error("[ParamsManager] Ключ не найден: " + item + " в пути: " + path);
            }

            if (ss.eof()) {
                if constexpr (std::is_same_v<T, cv::Size>) {
                    if (!it->second.template is<YamlArray>()) {
                        throw std::runtime_error("[ParamsManager] Ключ '" + item + "' не является массивом для cv::Size");
                    }
                    const YamlArray& arr = it->second.template get<YamlArray>();
                    if (arr.size() != 2) {
                        throw std::runtime_error("[ParamsManager] Массив '" + item + "' должен содержать ровно 2 элемента для cv::Size");
                    }
                    return cv::Size(arr[0].template get<int>(), arr[1].template get<int>());
                }

                else if constexpr (std::is_same_v<T, bool>) {
                    if (it->second.template is<int>()) {
                        return it->second.template get<int>() != 0;
                    }
                }
                
                return it->second.template get<T>();
            }

            if (!it->second.template is<YamlMap>()) {
                throw std::runtime_error("[ParamsManager] Узел не является секцией: " + item);
            }
            current_map = &it->second.template get<YamlMap>();
        }
        throw std::runtime_error("[ParamsManager] Неверный формат пути: " + path);
    }
};