#include "params_manager.hpp"
#include <map>

int main() {
    auto &manager = ParamsManager::getInstance();
    manager.load("../.config/params.yml");
    int &a = manager["test"]["a"];
    std::cout << a << std::endl;
    std::cin.get();
    manager.refresh();
    std::cout << a << std::endl;
    return 0;
}
