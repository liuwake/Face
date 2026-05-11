#include <iostream>
#include <inspireface.h>

int main(int argc, char* argv[]){
    if (argc < 2) {
        std::cerr << "Usage: ./demo <model_path>" << std::endl;
        return 1;
    }
    std::cout << "[SUCCESS] THE BINARY STARTED WITHOUT OPENCV CONFLICTS!" << std::endl;
    
    HPath packPath = argv[1];
    std::cout << "[DEBUG] Attempting to Launch SDK..." << std::endl;
    
    HResult ret = HFLaunchInspireFace(packPath);
    if (ret != 0) {
        std::cout << "[DEBUG] SDK Launch Failed with code: " << ret << std::endl;
        return 1;
    }
    
    std::cout << "[SUCCESS] SDK INITIALIZED SUCCESSFULLY ON TENSORRT!" << std::endl;
    return 0;
}