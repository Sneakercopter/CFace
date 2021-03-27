#ifndef DATECOPTER_EXTRACTFACES_H
#define DLIB_JPEG_SUPPORT
#define DATECOPTER_EXTRACTFACES_H

#include <iostream>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <iostream>

#include <opencv2/opencv.hpp>

using namespace dlib;
using namespace std;
using namespace cv;


class ExtractFaces {
public:
    ExtractFaces();
    cv::Mat cropFace(const string fileName, int width, int height);

private:
    frontal_face_detector detector;
    shape_predictor predictor;
    void showImage(Mat rect);
    cv::Rect dlibRectangleToOpenCV(dlib::rectangle r, int imgRows, int imgCols);
    dlib::rectangle openCVRectToDlib(cv::Rect r);
};


#endif //DATECOPTER_EXTRACTFACES_H
