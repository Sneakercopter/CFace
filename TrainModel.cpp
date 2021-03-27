#include "TrainModel.h"

TrainModel::TrainModel(cppflow::model model, string datasetPath) : model(model) {
    DataPrepare prep;
    this->yesImages = prep.getYesImagesInDir(datasetPath);
    this->noImages = prep.getNoImagesInDir(datasetPath);
}

// Run the training model
void TrainModel::runTraining(string saveDir, int epochs, int stepsPerEpoch, int batchSize){
    for(int i = 0; i < epochs; i++){
        for(int j = 0; j < stepsPerEpoch; j++){
            cout << "Training step epoch: " << i << "/" << epochs << ", step: " << j << "/" << stepsPerEpoch << endl;
            trainingStep(batchSize);
            predictionStep(batchSize);
            //break;
        }
        this->model.saveModel(saveDir);
        //break;
    }
    cout << "Training completed." << endl;
}

// Run a singular training step of the predefined batch size batch size by getting an image batch and passing it through the
// network and presenting the current loss.
void TrainModel::trainingStep(int batchSize) {
    std::vector<cppflow::tensor> imageVector;
    std::vector<float> groundTruth;
    getNewBatch(&imageVector, &groundTruth, batchSize);

    auto inputImages = cppflow::concat(cppflow::tensor(0), imageVector);

    int64 dimSize = groundTruth.size() / 2;
    const std::vector<std::int64_t> dims = {dimSize, 2};
    const auto data_size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<std::int64_t>{});

    float gTruth[groundTruth.size()];
    for(int i = 0; i < groundTruth.size(); i++){
        gTruth[i] = groundTruth[i];
    }

    auto inputTruthTensor = TF_AllocateTensor(TF_FLOAT, dims.data(), static_cast<int>(dims.size()), data_size);
    memcpy(TF_TensorData(inputTruthTensor), gTruth, sizeof(float) * groundTruth.size());
    auto inputTruth = cppflow::tensor(inputTruthTensor);

    auto loss = this->model(inputImages, inputTruth);

    auto lossVal = static_cast<float_t*>(TF_TensorData(loss.get_tensor().get()));
    cout << "Loss: " << lossVal[0] << endl;
}

// Run a single prediction step of the specified batch size by getting an image batch and passing it through the
// network and comparing it to the ground truth.
void TrainModel::predictionStep(int batchSize) {
    std::vector<cppflow::tensor> imageVector;
    std::vector<float> groundTruth;
    getNewBatch(&imageVector, &groundTruth, batchSize);

    auto inputImages = cppflow::concat(cppflow::tensor(0), imageVector);

    auto predictions = this->model(inputImages);
    auto decisions = cppflow::arg_max(predictions, 1);
    auto decisionArr = static_cast<int*>(TF_TensorData(decisions.get_tensor().get()));

    int correct = 0;
    int predictionCounter = 0;

    cout << "Prediction = ";
    for(int i = 0; i < groundTruth.size()-1; i = i+2){
        float no = groundTruth[i];
        float yes = groundTruth[i+1];
        int prediction = decisionArr[predictionCounter];
        predictionCounter++;
        cout << prediction << " ";

        if(no > yes){
            if(prediction == 0){
                correct++;
            }
        }
        else if(yes > no){
            if(prediction == 1){
                correct++;
            }
        }
    }
    cout << endl;
    float accuracy = ((float)correct / (float)batchSize) * 100.0;

    cout << "Accuracy: " << accuracy << "%" << endl;
}

// Format batches to the specified size and assign them to the image vector pointer while also keeping track of the
// ground truth for later accuracy analysis.
void TrainModel::getNewBatch(std::vector<cppflow::tensor> *imageVector, std::vector<float> *groundTruth, int batchSize){
    DataPrepare dataPrep;
    int batchCount = 0;
    for(batchCount; batchCount < batchSize; batchCount++){
        int yesOrNo = std::rand() % 2 + 1;;
        if(yesOrNo == 1){
            int selection = std::rand() % this->yesImages.size() + 0;
            auto input = cppflow::decode_jpeg(cppflow::read_file((string)this->yesImages[selection].c_str()));
            input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
            input = input / 255.f;
            input = cppflow::expand_dims(input, 0);
            imageVector->push_back(input);
            groundTruth->push_back(0.0);
            groundTruth->push_back(1.0);
        }
        else{
            int selection = std::rand() % this->noImages.size() + 0;
            auto input = cppflow::decode_jpeg(cppflow::read_file((string)this->noImages[selection].c_str()));
            input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
            input = input / 255.f;
            input = cppflow::expand_dims(input, 0);
            imageVector->push_back(input);
            groundTruth->push_back(1.0);
            groundTruth->push_back(0.0);
        }
    }
}

