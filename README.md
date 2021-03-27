# CFace
**Training and Evaluating Facial Classification Keras Models using the Tensorflow C API Implemented into a C++ Codebase.**

## Dependancies
- [Tensorflow 2.0+ For Python](https://www.tensorflow.org/install)
- [Tensorflow C API](https://www.tensorflow.org/install/lang_c "Tensorflow C API")
- [DLIB for C++](http://dlib.net/compile.html)
- [OpenCV for C++](https://opencv.org/releases/)
- [Boost](https://stackoverflow.com/questions/3897839/how-to-link-c-program-with-boost-using-cmake)
- Custom version of [CPPFlow](https://github.com/serizba/cppflow "CPPFlow") (Included in Repository)

## Features
CFace has the ability to implement transfer learning into C++ based applications with Keras pretrained models. The implementation presented here is a binary classifier between two image categories. CFace has the ability to preprocess facial images through the use of facial detection, crop them to the desired model input size and store the processed file into a file structure ready for training.

It is also possible to do full training cycles using CFace with strictly only calls from the Tensorflow C API which is seldomly documented with a complete and comprehensive example. The model required for training is created and exported through a simple Python script which compiles the graph structure with custom overridden prediction and training functionality which is then exported into a format where weights and structure can be easily accessed and modified by C++ code.

Training allows for custom batch sizes, accuracy and loss tracking and user defined epoch cycles allowing for shorter and longer training sessions based on user requirement. Modification of input shapes is also possible to fit the needs of your model. Once training is completed, model weights are also exported directly from the C++ code with no further intervention from Python or Keras needed.

## Building
This repository includes a `CMakeLists.txt` file which allows for the building of this project, it is recommended to do so in the `release` mode in order to make the executable as efficient as possible.

The included code was run and tested on a Macbook running OSX 11.1 and modification will likely be needed to build for other operating systems and or individual systems!

## Creating and Exporting the Keras Model
In the ``ModelCreation`` folder in the root directory there are two Python files that create and export the required Keras model.
CFace uses Transfer Learning for it's feature extraction layer and then only has learnable parameters in it's dense layers.
This is done in this way to speed up the training time and also to leverage more complex models that have been trained on large datasets in a smaller context.

To export your own "untrainedModel" in the correct format ready for CFace, simply execute `main.py` with Python and the model folder titled `untrainedModel` will be exported to the execution directory.
This entire folder includes the graph structure of the model along with the current weights needed for further training. It is important to keep the folder in this structure as CFace will need this format in order to load the model.

By default this will export the model with an input Tensor shape of `(?, 71, 71, 3)` and an output Tensor shape of `(?, 2)`.

## Input Image Folder Structure
CFace has a predefined folder structure for image preprocessing and image training. CFace has the ability to automatically detect faces within photos, crop them to the appropriate size needed by the input layer of the Tensorflow model and store them in the correct format for training.
This means that the first step for any training should be to preprocess the images, but this also requires a defined folder structure. This is outlined by the structure below.
```
- TrainingImages
| -- Yes
| | -- YesImageCategory1.jpeg
| | -- YesImageCategory2.jpeg
| | -- TheseNamesCanBeAnything.jpeg
| -- No
| | -- NoImageCategory1.jpeg
| | -- NoImageCategory2.jpeg
| | -- TheseNamesCanAlsoBeAnything.jpeg
```
What is important to note is that as this is a binary classifier, the two classes have by default been named `Yes` and `No` and two folders by these names (Case Sensitive).
These two folders will hold the images for each category respectively. The `Yes` and `No` folder will need to live in a parent folder called `TrainingImages`.
This parent folder is what will be passed to the C++ code moving forward.

## Running Image Preprocessing
Running the preprocessing of images is easy and an example is presented in `main.cpp`, another example is provided in the code below.
```c++
#include <stdlib.h>
#include "lib/DataPrepare.h"

using namespace std;
using namespace boost::filesystem;

int main() {
    // Silence Tensorflow logging
    char* new_environment = "TF_CPP_MIN_LOG_LEVEL=3";
    putenv(new_environment);
    // Image cropping and preparation
    DataPrepare prep;
    prep.cropFacesInDir("/full/path/to/project/TrainingImages");
    return 0;
}
```
This code will first identify if the provided parent directory has the correct folder structure as defined above, and if so read out all valid `jpeg` image files.
These images are then passed to `ExtractFaces.cpp` which leverages the `frontal_face_detector` from DLIB. It is important to note that for every photo only the first face found will be cropped and categorised.
If no face is found at all, the image is skipped. This facial detector is not perfect and if you want to implement your own algorithm you can modify `ExtractFace.cpp` to do so.

Once all identified faces are identified, they are cropped and saved to the a new directory within the root directory passed to `prep.cropFacesInDir` in the following format.
```
- TrainingImages
| -- cropped
| | -- Yes
| | | -- CroppedYesImageCategory1.jpeg
| | | -- CroppedYesImageCategory2.jpeg
| | | -- CroppedTheseNamesCanBeAnything.jpeg
| | -- No
| | | -- CroppedNoImageCategory1.jpeg
| | | -- CroppedNoImageCategory2.jpeg
| | | -- CroppedTheseNamesCanAlsoBeAnything.jpeg
```
This `cropped` directory is already in the correct folder and file structure format and does not need to be modified. The path to this cropped directory is what will need to be passed to the training/execution steps of CFace.

## Running Model Training
Training models with CFace is abstracted to be very simple. To run training simply, simple add the following lines to the previous Image Preprocessing codebase.
```c++
#include <stdlib.h>
#include "lib/DataPrepare.h"
#include "lib/TrainModel.h"
#include "cppflow/model.h"

using namespace std;
using namespace boost::filesystem;

int main() {
    // Silence Tensorflow logging
    char* new_environment = "TF_CPP_MIN_LOG_LEVEL=3";
    putenv(new_environment);
    // Image cropping and preparation
    DataPrepare prep;
    prep.cropFacesInDir("/full/path/to/project/TrainingImages");
    // Defining training parameters
    int epochs = 100;
    int stepsPerEpoch = 39;
    int batchSize = 8;
    // Training models
    TrainModel trainer(cppflow::model("/full/path/to/project/untrainedModel"),
                       "/full/path/to/project/TrainingImages/cropped");
    trainer.runTraining("/full/path/to/project/TrainedWeights",
                        epochs,
                        stepsPerEpoch,
                        batchSize);
    return 0;
}
```
This code will take the model exported from Python Keras and saved to `/full/path/to/project/untrainedModel` and train it against the preprocessed images and categories located at  `"/full/path/to/project/TrainingImages/cropped"`.
Once completed, the newly trained weights will be exported to `/full/path/to/project/TrainedWeights`.

`trainer.runTraining` takes training parameters, their function is described as follows.
- `epochs`: Amount of times `stepsPerEpoch` is iterated.
- `stepsPerEpoch`: Amount of training steps we execute per epoch.
- `batchSize`: Amount of images to be passed through the network per training step.

There is no "best" configuration for these parameters, they are problem dependant and need to be figured out through testing.

## Implementation Notes
In order to be as useful as possible, a lot of the heavy interaction with the Tensorflow C API is abstracted away.
There is some complex interaction with Tensorflow sessions required in order to achieve the result above and can be dived into and modified.
There are also custom modifications made to the [CPPFlow](https://github.com/serizba/cppflow "CPPFlow") library included with this file. All custom code and modifications are outlined in the list below.
- `lib/DataPrepare.cpp`
- `lib/DataPrepare.h`
- `lib/ExtractFaces.cpp`
- `lib/ExtractFaces.h`
- `lib/TrainModel.cpp`
- `lib/TrainModel.h`
- `lib/main.cpp`
- `cppflow/include/cppflow/model.h` (Modified)

## Future ToDo
- Improve code comments and documentation
- Add explicit model execution API
- Make it easier to change 
- Document build examples for systems outside of CMake
- Implement comprehensive testing
- Clean up code memory leaks that may be present
- Add functionality for more than 2 classification categories
