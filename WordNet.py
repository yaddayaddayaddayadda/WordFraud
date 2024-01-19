from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import numpy as np

np.random.seed(1337)
tf.random.set_seed(1337)
class WordNet:
    def __init__(self, width, height, depth, no_classes, no_filters, kernel_size, no_convlayers, pool_size, no_fclayers, fc_size):
        self.width = width
        self.height = height
        self.depth = depth
        self.input_shape = (height, width, depth)
        self.no_classes = no_classes
        self.no_filters = no_filters
        self.kernel_size = kernel_size
        self.no_convlayers = no_convlayers
        self.pool_size = pool_size
        self.no_fclayers = no_fclayers
        self.fc_size = fc_size
        self.model = Sequential()
    
    def build_model(self):
        for layer in range(self.no_convlayers):
            if layer == 0:
                self.model.add(Conv2D(self.no_filters, self.kernel_size, padding="same", input_shape=self.input_shape))
                self.model.add(BatchNormalization())
                self.model.add(Activation("relu"))
                self.model.add(MaxPooling2D(pool_size=self.pool_size))
            else:
                self.model.add(Conv2D(self.no_filters, self.kernel_size, padding="same"))
                self.model.add(BatchNormalization())
                self.model.add(Activation("relu"))
                self.model.add(MaxPooling2D(pool_size=self.pool_size))
        for layer in range(self.no_fclayers):
            if layer == 0:
                self.model.add(Flatten())
                self.model.add(Dense(self.fc_size))
                self.model.add(Activation("relu"))
                self.model.add(Dropout(0.5))
            else:
                self.model.add(Dense(self.fc_size))
                self.model.add(Activation("relu"))
                self.model.add(Dropout(0.5))
        # softmax classifier
        self.model.add(Dense(self.no_classes))
        if self.no_classes == 2:
            self.model.add(Activation("sigmoid"))
        else:
            self.model.add(Activation("softmax"))
        # return the constructed network architecture
        return self.model