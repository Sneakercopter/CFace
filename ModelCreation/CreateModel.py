from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf

class MyModel(tf.keras.Model):

    def __init__(self, name=None, numClasses=2, imageHeightWidth=71, imageChannels=3, **kwargs):
        super().__init__(**kwargs)
        # Here we are loading the predefined Xception layer with pretrained weights from 'imagenet'. You can change this
        # to any model you see fit (or even build your own convolutional network for training).
        self.vgModel = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(imageHeightWidth, imageHeightWidth, imageChannels))
        # We are doing transfer learning, therefore we declare the pretrained model layers untrainable
        for layer in self.vgModel.layers:
            layer.trainable = False
        self.model = Sequential()
        self.model.add(self.vgModel)
        # Now we add the trainable dense layers to relearn new categorization of extracted features
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Flatten())
        # Final output layer must output the number of classes as this will be where predictions per
        # class are made
        self.model.add(Dense(numClasses, activation='softmax'))
        # Finally, mark the first (pretrained) layer as untrainable in the Sequential model scope as well
        self.model.layers[0].trainable = False

    # This is the TF function we override in order to call prediction from the C API directly
    @tf.function
    def call(self, inputs, training=False):
        return self.model(inputs)

    # This is the TF function we override in order to enable training and loss extraction from the C API directly
    @tf.function
    def training(self, data):
        loss = self.train_step(data)['loss']
        return {'loss': loss}
