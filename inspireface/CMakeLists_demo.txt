cmake_minimum_required(VERSION 3.10)
project(InspireFaceDemo)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV components
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)

# Define paths
set(INSPIRE_FACE_ROOT "/kaggle/temp/InspireFace")
set(INSPIRE_FACE_LIB_DIR "${INSPIRE_FACE_ROOT}/build/inspireface-linux/InspireFace/lib")

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    "${INSPIRE_FACE_ROOT}/3rdparty/MNN/include"
    "${INSPIRE_FACE_ROOT}/3rdparty/InspireCV/include"
    "${INSPIRE_FACE_ROOT}/cpp/inspireface/include"
    "${INSPIRE_FACE_ROOT}/cpp/inspireface/c_api"
)

# Link directories
link_directories(${INSPIRE_FACE_LIB_DIR})

# Add executable
add_executable(demo demo.cpp)

# Link libraries
target_link_libraries(demo
    ${OpenCV_LIBS}
    InspireFace
)
