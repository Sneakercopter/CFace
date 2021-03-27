#include "ExtractFaces.h"

ExtractFaces::ExtractFaces() {
    this->detector = get_frontal_face_detector();
}

// Convert between the two rectangle types
//
cv::Rect ExtractFaces::dlibRectangleToOpenCV(dlib::rectangle r, int imgRows, int imgCols) {
    int left = max((int)r.left(), 0);
    int top = max((int)r.top(), 0);
    int right = min((int)r.right() + 1, imgCols);
    int bottom = min((int)r.bottom() + 1, imgRows);

    return cv::Rect(cv::Point2i(left, top), cv::Point2i(right, bottom));
}

dlib::rectangle ExtractFaces::openCVRectToDlib(cv::Rect r) {
    long tlX = (long)max(r.tl().x, 0);
    long tlY = (long)max(r.tl().y, 0);
    long brX = (long)max(r.br().x - 1, 0);
    long brY = (long)max(r.br().y - 1, 0);
    return dlib::rectangle(tlX, tlY, brX, brY);
}

// Preview the image if one would like
void ExtractFaces::showImage(Mat rect) {
    String windowName = "showImage Window";
    namedWindow(windowName);
    imshow(windowName, rect);
    // Wait for any keystroke in the window
    waitKey(0);
    destroyWindow(windowName);
}

// Open the filename and load it as a matrix. The matrix is passed to facial detection and is then cropped to only
// include the face and then finally resized to the wanted width and height. Returned as a Matrix.
cv::Mat ExtractFaces::cropFace(const string fileName, int width, int height) {
    array2d<unsigned char> img;

    // Here we process the image with opencv2
    cv::Mat croppedImage;
    Mat readImg;
    Mat resizedImg;
    readImg = imread( fileName, 1 );
    cv::resize(readImg, resizedImg, cv::Size(), 1.0, 1.0);

    // Keep track of these to make sure our resizing of dimensions does not mess up
    int imgRows = resizedImg.rows;
    int imgCols = resizedImg.cols;

    // Now the image is converted to dlib format and detector is run
    cv_image<bgr_pixel> cimg(resizedImg);
    std::vector<dlib::rectangle> detectedFaces = detector(cimg, 1);
    if(!detectedFaces.empty()){
        // Pull out the first face
        dlib::rectangle elm = detectedFaces[0];

        // Convert between the two rectangle types
        cv::Rect cropped = dlibRectangleToOpenCV(elm, imgRows, imgCols);
        croppedImage = resizedImg(cropped);

        Mat finalImg;
        cv::resize(croppedImage, finalImg, cv::Size(width, height));
        // Return the resized and cropped image
        return finalImg;
    }
    else{
        // We could not find a face so return the null image
        return croppedImage;
    }
}