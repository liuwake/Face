cd /kaggle/temp/InspireFace
g++ -std=c++14 -O3 /kaggle/temp/Face/inspireface/demo.cpp \
    -I./3rdparty/MNN/include \
    -I./3rdparty/InspireCV/include \
    -I./cpp/inspireface/include \
    -I./cpp/inspireface/c_api \
    -I/usr/include/opencv4 \
    -L./build/inspireface-linux/install/lib \
    -lInspireFace \
    -lopencv_core -lopencv_imgproc -lopencv_highgui \
    -o demo