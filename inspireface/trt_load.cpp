#include <iostream>
#include <inspireface.h>

int main() {
    std::cout << "[SUCCESS] THE BINARY STARTED WITHOUT OPENCV CONFLICTS!" << std::endl;
    
    HPath packPath = "/kaggle/temp/InspireFace/test_res/pack/Megatron_TRT";
    std::cout << "[DEBUG] Attempting to Launch SDK..." << std::endl;
    
    HResult ret = HFLaunchInspireFace(packPath);
    if (ret != 0) {
        std::cout << "[DEBUG] SDK Launch Failed with code: " << ret << std::endl;
        return 1;
    }
    
    std::cout << "[SUCCESS] SDK INITIALIZED SUCCESSFULLY ON TENSORRT!" << std::endl;
    return 0;
}