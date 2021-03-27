#include "DataPrepare.h"

DataPrepare::DataPrepare() {
    ;
}

// Validates folder structure and ensures all valid file type paths are returned in a vector
//
std::vector<path> DataPrepare::getImagesInDir(string dirName){
    std::vector<path> allPaths;
    if(!exists(dirName + "/Yes")){
        cout << "The \"Yes\" classification directory is missing from " << dirName << "." << endl;
        return allPaths;
    }
    else if(!exists(dirName + "/No")){
        cout << "The \"No\" classification directory is missing from " << dirName << "." << endl;
        return allPaths;
    }
    else{
        cout << "Classification folders have been found and validated." << endl;
        path yesPath(dirName + "/Yes");
        path noPath(dirName + "/No");
        directory_iterator directoryIterator;
        for ( directory_iterator itr(yesPath);  itr != directoryIterator; ++itr) {
            path currPath = itr->path();
            if(boost::algorithm::ends_with(currPath.filename().c_str(), ".jpeg")){
                allPaths.push_back(currPath);
            }
            else{
                cout << "Ignoring non-image file: " << currPath.filename() << endl;
            }
        }
        for (directory_iterator itr(noPath);  itr != directoryIterator; ++itr) {
            path currPath = itr->path();
            if(boost::algorithm::ends_with(currPath.filename().c_str(), ".jpeg")){
                allPaths.push_back(currPath);
            }
            else{
                cout << "Ignoring non-image file: " << currPath.filename() << endl;
            }
        }
        cout << "Identified " << allPaths.size() << " images." << endl;
        return allPaths;
    }
}

// Pull only the valid images in the "Yes" category directory (If it exists) and return it in a vector.
std::vector<path> DataPrepare::getYesImagesInDir(string dirName){
    std::vector<path> yesPaths;
    if(!exists(dirName + "/Yes")){
        cout << "The \"Yes\" classification directory is missing from " << dirName << "." << endl;
        return yesPaths;
    }
    path yesPath(dirName + "/Yes");
    directory_iterator directoryIterator;
    for ( directory_iterator itr(yesPath);  itr != directoryIterator; ++itr) {
        path currPath = itr->path();
        if(boost::algorithm::ends_with(currPath.filename().c_str(), ".jpeg")){
            yesPaths.push_back(currPath);
        }
        else{
            cout << "Ignoring non-image file: " << currPath.filename() << endl;
        }
    }
    return yesPaths;
}

// Pull only the valid images in the "No" category directory (If it exists) and return it in a vector.
std::vector<path> DataPrepare::getNoImagesInDir(string dirName){
    std::vector<path> noPaths;
    if(!exists(dirName + "/No")){
        cout << "The \"No\" classification directory is missing from " << dirName << "." << endl;
        return noPaths;
    }
    path noPath(dirName + "/No");
    directory_iterator directoryIterator;
    for ( directory_iterator itr(noPath);  itr != directoryIterator; ++itr) {
        path currPath = itr->path();
        if(boost::algorithm::ends_with(currPath.filename().c_str(), ".jpeg")){
            noPaths.push_back(currPath);
        }
        else{
            cout << "Ignoring non-image file: " << currPath.filename() << endl;
        }
    }
    return noPaths;
}

// Resize all images within a directory name to the defined width and height.
void DataPrepare::resizeImagesInDir(string dirName, int height, int width){
    std::vector<path> imagePaths = getImagesInDir(dirName);
    for(path curr : imagePaths){
        string parentPath = curr.parent_path().c_str();
        string fileName = curr.filename().c_str();
        string fullpath = parentPath + "/" + fileName;
        Mat readImg;
        readImg = imread( fullpath, 1 );
        Mat finalImg;
        cv::resize(readImg, finalImg, cv::Size(width, height));
        cout << "Resized: " << fullpath << endl;
        cv::imwrite(fullpath, finalImg);
    }
}

// Crop all images within a defined directory name to the required input width and height ready for the training model
void DataPrepare::cropFacesInDir(string dirName){
    int facesFound = 0;
    int imagesProcessed = 0;
    int width = 71;
    int height = 71;
    path croppedPath = path(dirName + "/cropped");
    path croppedYesPath = path(dirName + "/cropped/Yes");
    path croppedNoPath = path(dirName + "/cropped/No");
    if(!exists(croppedPath)){
        cout << "No cropped path exists, creating a new one..." << endl;
        create_directory(croppedPath);
        create_directory(croppedYesPath);
        create_directory(croppedNoPath);
    }

    std::vector<path> imagePaths = getImagesInDir(dirName);
    for(path curr : imagePaths){
        imagesProcessed++;
        string parentPath = curr.parent_path().c_str();
        string fileName = curr.filename().c_str();
        string fullpath = parentPath + "/" + fileName;
        cv::Mat croppedMatrix = this->faceExtraction.cropFace(fullpath, width, height);
        if(croppedMatrix.dims > 0){
            facesFound++;
            // This means we have a valid cropped face that we can save
            if(boost::algorithm::ends_with(parentPath, "/Yes")){
                string savePath = (string)croppedYesPath.c_str() + "/" + fileName;
                cv::imwrite(savePath, croppedMatrix);
            }
            else if(boost::algorithm::ends_with(parentPath, "/No")){
                string savePath = (string)croppedNoPath.c_str() + "/" + fileName;
                cv::imwrite(savePath, croppedMatrix);
            }
            else{
                cout << "Unknown save path destination: " << parentPath << endl;
            }
        }
        cout << "Faces found: " << facesFound << ", total images processed " << imagesProcessed << endl;
    }
}