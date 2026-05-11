#include <iostream>

#include <opencv2/opencv.hpp>

#include <inspireface.h>
#include <herror.h>


int main(int argc, char* argv[]){
    if (argc < 4) {
        std::cerr << "Usage: ./demo <model_path> <image_path> <output_path>" << std::endl;
        return 1;
    }
    HResult ret;
    // The resource file must be loaded before it can be used
    HPath packPath = argv[1];
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
    HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
// Maximum number of faces detected
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
    HFSessionSetFilterMinimumFacePixelSize(session, 4);

// Load a image
    HFImageBitmap image;
    HPath sourcePath = argv[2];
    ret = HFCreateImageBitmapFromFilePath(sourcePath, 3, &image);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "The source entered is not a picture or read error.");
        return ret;
    }
// Prepare an image parameter structure for configuration
    HFImageStream imageHandle = {0};
    HFRotation rotation_enum = HF_CAMERA_ROTATION_0;
    ret = HFCreateImageStreamFromImageBitmap(image, rotation_enum, &imageHandle);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create ImageStream error: %d", ret);
        return ret;
    }

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

    cv::Mat img = cv::imread(sourcePath);
    for (int i = 0; i < faceNum; i++) {
        cv::Rect rect;
        rect.x = multipleFaceData.rects[i].x;
        rect.y = multipleFaceData.rects[i].y;
        rect.width = multipleFaceData.rects[i].width;
        rect.height = multipleFaceData.rects[i].height;
        cv::rectangle(img, rect, {255, 0, 0}, 2);
    }
    cv::imwrite(argv[3], img);
    HFLogPrint(HF_LOG_INFO, "saved img at %s", argv[3]);

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