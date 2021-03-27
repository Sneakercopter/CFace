#include <stdlib.h>

#include "DataPrepare.h"
#include "TrainModel.h"
#include "cppflow/model.h"

using namespace std;
using namespace boost::filesystem;

int main() {
    // Silence Tensorflow logging
    char* new_environment = "TF_CPP_MIN_LOG_LEVEL=3";
    putenv(new_environment);

    // Preparing and identifying local directories
    //
    // Check for pretrained models, ensure untrained model copy is available
    // create folders if they do not yet exist
    //
    DataPrepare prep;
    prep.cropFacesInDir("/full/path/to/project/TrainingImages");

    // Tensorflow training
    //
    int epochs = 1000;
    int stepsPerEpoch = 39;
    int batchSize = 8;

    TrainModel t(cppflow::model("/full/path/to/project/untrainedModel"),
                 "/full/path/to/project/TrainingImages/cropped");
    t.runTraining("/full/path/to/project/TrainedWeights", epochs, stepsPerEpoch, batchSize);

    return 0;
}
