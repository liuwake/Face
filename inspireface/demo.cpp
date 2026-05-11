#include <iostream>
#include <inspireface.h>
#include <opencv2/opencv.hpp>
#include <herror.h>


int main(int argc, char* argv[]){
    if (argc < 3) {
        std::cerr << "Usage: ./demo <model_path> <image_path>" << std::endl;
        return 1;
    }
    HResult ret;
    HPath packPath = argv[1];
    ret = HFLaunchInspireFace(packPath);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Load Resource error: %d", ret);                             
        return ret;
    }
    HOption option = HF_ENABLE_QUALITY | HF_ENABLE_MASK_DETECT | HF_ENABLE_LIVENESS;
    HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;

    HInt32 maxDetectNum = 20;
    HInt32 detectPixelLevel = 160;
    HFSession session = {0};
    ret = HFCreateInspireFaceSessionOptional(option, detMode, maxDetectNum, detectPixelLevel, -1, &session);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create FaceContext error: %d", ret);
        return ret;
    }

    HFSessionSetTrackPreviewSize(session, detectPixelLevel);
    HFSessionSetFilterMinimumFacePixelSize(session, 4);


    HFImageBitmap image;
    HPath sourcePath = argv[2];
    ret = HFCreateImageBitmapFromFilePath(sourcePath, 3, &image);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "The source entered is not a picture or read error.");
        return ret;
    }

    HFImageStream imageHandle = {0};
    HFRotation rotation_enum = HF_CAMERA_ROTATION_0;
    ret = HFCreateImageStreamFromImageBitmap(image, rotation_enum, &imageHandle);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create ImageStream error: %d", ret);
        return ret;
    }

    HFMultipleFaceData multipleFaceData = {0};
    ret = HFExecuteFaceTrack(session, imageHandle, &multipleFaceData);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Execute HFExecuteFaceTrack error: %d", ret);
        return ret;
    }

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

    for (int i = 0; i < faceNum; i++) {
        std::cout << "Face " << i << ": "
                  << "x=" << multipleFaceData.rects[i].x
                  << ", y=" << multipleFaceData.rects[i].y
                  << ", w=" << multipleFaceData.rects[i].width
                  << ", h=" << multipleFaceData.rects[i].height
                  << ", roll=" << multipleFaceData.angles.roll[i]
                  << std::endl;
    }

    cv::Mat img = cv::imread(sourcePath);
    for (int i = 0; i < faceNum; i++) {
        cv::Rect rect;
        rect.x = multipleFaceData.rects[i].x;
        rect.y = multipleFaceData.rects[i].y;
        rect.width = multipleFaceData.rects[i].width;
        rect.height = multipleFaceData.rects[i].height;
        cv::rectangle(img, rect, {255, 0, 0}, 2);
    }
    cv::imwrite("output.jpg", img);

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