from keras.optimizers import RMSprop
import tensorflow as tf

from CreateModel import MyModel

# Amount of classes we want to predict (2 for a binary classifier as in this example)
numClasses = 2
# We usually crop to squares, so height and width can be the same
imageHeightWidth = 71
# We are using color images, so we have 3 color channels (RGB).
# Change to 1 for greyscale or 2/4 if you also include alpha channels.
imageChannels = 3

customModel = MyModel(name="end_model", numClasses=numClasses, imageHeightWidth=imageHeightWidth, imageChannels=imageChannels)

# When compiling the model we are defining our loss function and optimizer/learning rate. You can
# change these to how you see fit.
customModel.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001))

# These concrete functions are exported into graph operations with the following functions. They will later be compiled
# into the exported model but serve the purpose of exposing the TF.function overrides we made in CreateModel.MyModel
# to the C API by declaring them as graph operations. It is important that the TensorSpecs match the input Matrices we
# use in the C++ code as we will encounter bugs if not. They also must match in terms of TF types. It is also important
# to note that "None" in this model shape context refers to a wildcard which will allow any amount of inputs of the
# shape - this is to allow for varying batch sizes during training and prediction.
train_output = customModel.training.get_concrete_function((tf.TensorSpec([None, imageHeightWidth, imageHeightWidth, imageChannels], tf.float32, name='inputs'), tf.TensorSpec([None,numClasses], tf.float32, name='target')))
call_output = customModel.call.get_concrete_function((tf.TensorSpec([None, imageHeightWidth, imageHeightWidth, imageChannels], tf.float32, name='inputs')))

# This will save the model and pretrained weights in the correct format ready to be used in C API training
# calls. Important notes about this function is the save format and signatures. The save format must be explicitly
# defined as "tf" so that we have the folder structure correct with separated weights and model structure.
# The signatures are the additional overridden concrete functions we have defined which we will be calling in the graph
# operations from C API and must keep the given names.
# (You may change them but you need to modify the references in the C++ code as well)
customModel.save("untrainedModel", save_format="tf", signatures={'train': train_output, 'predict': call_output})
