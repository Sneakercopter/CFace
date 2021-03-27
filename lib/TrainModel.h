#ifndef DATECOPTER_TRAINMODEL_H
#define DATECOPTER_TRAINMODEL_H

#include <iostream>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "DataPrepare.h"

#include <tensorflow/c/c_api.h>

#include "cppflow/ops.h"
#include "cppflow/model.h"

using namespace std;
using namespace boost::filesystem;


class TrainModel {
public:
    TrainModel(cppflow::model model, string datasetPath);
    void runTraining(string saveDir, int epochs, int stepsPerEpoch, int batchSize);
    void trainingStep(int batchSize);
    void predictionStep(int batchSize);
    void getNewBatch(std::vector<cppflow::tensor> *imageVector, std::vector<float> *groundTruth, int batchSize);

    string filename;
    cppflow::model model;
    std::vector<path> yesImages;
    std::vector<path> noImages;

private:
};


#endif //DATECOPTER_TRAINMODEL_H
