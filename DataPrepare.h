#ifndef DATECOPTER_DATAPREPARE_H
#define DATECOPTER_DATAPREPARE_H

#include <iostream>
#include <filesystem>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "ExtractFaces.h"

using namespace std;
using namespace boost::filesystem;

class DataPrepare {
public:
    ExtractFaces faceExtraction;

    DataPrepare();
    void cropFacesInDir(string dirName);
    void resizeImagesInDir(string dirName, int height, int width);
    std::vector<path> getImagesInDir(string dirName);
    std::vector<path> getYesImagesInDir(string dirName);
    std::vector<path> getNoImagesInDir(string dirName);
};


#endif //DATECOPTER_DATAPREPARE_H
