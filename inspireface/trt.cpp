#include <iostream>
#include <inspireface.h>
#include <opencv2/opencv.hpp>




int main(){
    HResult ret;
    // TODO
    // # HPath packPath = "/kaggle/input/Megatron"; // OK
    HPath packPath = "/kaggle/input/Megatron_TRT"; // TRT NG
    std::cout << "[DEBUG] Attempting to Launch SDK..." << std::endl;
// The resource file must be loaded before it can be used
    ret = HFLaunchInspireFace(packPath);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Load Resource error: %d", ret);                             
        return ret;
    }
// Enable the functions in the pipeline: mask detection, live detection, and face quality
// detection
    HOption option = HF_ENABLE_QUALITY | HF_ENABLE_MASK_DETECT | HF_ENABLE_LIVENESS;
// Non-video or frame sequence mode uses IMAGE-MODE, which is always face detection without
// tracking
    HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT; // 每次都重新检测所有人脸，不追踪，视频流有额外的模式
    //fixme:测试视频流模式
// Maximum number of faces detected 最大检测人脸数
    HInt32 maxDetectNum = 20;
// Face detection image input level
    HInt32 detectPixelLevel = 160;
// Handle of the current face SDK algorithm context
    HFSession session = {0};
    ret = HFCreateInspireFaceSessionOptional(option, detMode, maxDetectNum, detectPixelLevel, -1, &session);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create FaceContext error: %d", ret);
        return ret;
    }

// Configure some detection parameters
    HFSessionSetTrackPreviewSize(session, detectPixelLevel);
    HFSessionSetFilterMinimumFacePixelSize(session, 4); // 设置最小检测人脸像素尺寸
    std::cout << "Load a image" << std::endl;
// Load a image
    HFImageBitmap image;
    HPath sourcePath = "selected_img";
    ret = HFCreateImageBitmapFromFilePath(sourcePath, 3, &image);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "The source entered is not a picture or read error.");
        return ret;
    }
// Prepare an image parameter structure for configuration
    HFImageStream imageHandle = {0};
    HFRotation rotation_enum = HF_CAMERA_ROTATION_0; // 旋转角度，0表示不旋转，用于修正图像方向
    ret = HFCreateImageStreamFromImageBitmap(image, rotation_enum, &imageHandle);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create ImageStream error: %d", ret);
        return ret;
    }
    std::cout << "HFExecuteFaceTrack" << std::endl;
// Execute HF_FaceContextRunFaceTrack captures face information in an image
    HFMultipleFaceData multipleFaceData = {0};
    ret = HFExecuteFaceTrack(session, imageHandle, &multipleFaceData);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Execute HFExecuteFaceTrack error: %d", ret);
        return ret;
    }

// Print the number of faces detected
    auto faceNum = multipleFaceData.detectedNum;
    HFLogPrint(HF_LOG_INFO, "Num of face: %d", faceNum);

//    typedef struct HFMultipleFaceData {
//        HInt32 detectedNum;        ///< Number of faces detected.
//        PHFaceRect rects;          ///< Array of bounding rectangles for each face.
//        HPInt32 trackIds;          ///< Array of track IDs for each face.
//        HPInt32 trackCounts;       ///< Array of track counts for each face.
//        HPFloat detConfidence;     ///< Array of detection confidence for each face.
//        HFFaceEulerAngle angles;   ///< Euler angles for each face.
//        PHFFaceBasicToken tokens;  ///< Tokens associated with each face.
//    } HFMultipleFaceData, *PHFMultipleFaceData;

    if (faceNum > 0) {
        std::cout<<multipleFaceData.rects->x<<std::endl;
        std::cout<<multipleFaceData.rects->y<<std::endl;
        std::cout<<multipleFaceData.rects->width<<std::endl;
        std::cout<<multipleFaceData.rects->height<<std::endl;
        std::cout<<multipleFaceData.angles.roll<<std::endl;

        cv::Mat img = cv::imread("selected_img");
        if (!img.empty()) {
            cv::Rect rect;
            rect.height = multipleFaceData.rects->height;
            rect.width = multipleFaceData.rects->width;
            rect.x = multipleFaceData.rects->x;
            rect.y = multipleFaceData.rects->y;
            cv::rectangle(img,rect,{255,0,0},1);
            cv::imwrite("output.jpg", img);
            std::cout << "Output image written to output.jpg" << std::endl;
        } else {
            std::cerr << "Error: Could not read selected_img!" << std::endl;
        }
        std::cout << "98" << std::endl;
    } else {
        std::cout << "No faces detected!" << std::endl;
    }
// The memory must be freed at the end of the program
    ret = HFReleaseImageBitmap(image);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Release image bitmap error: %d", ret);
        return ret;
    }

    ret = HFReleaseImageStream(imageHandle);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Release image stream error: %d", ret);
    }

    ret = HFReleaseInspireFaceSession(session);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Release session error: %d", ret);
        return ret;
    }
}