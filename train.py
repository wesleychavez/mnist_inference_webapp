# train.py - Trains a CNN on MNIST and saves best model according to validation
# accuracy. A couple hyperparameters are located in config_train.py (I usually like
# to keep them separate), but model hyperparameters are here because they're static.
#
# This model achieves about 99% accuracy on the test set.
#
# This model takes about 45 minutes per epoch on a 2018 MacBook Pro and
#                  about 3 minutes per epoch on a AWS EC2 p2.xlarge
#
# - Wesley Chavez, 11/28/18

from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Concatenate, Flatten
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

import config_train as config

def create_model():
    inputs = Input(shape=(28,28,1))

    # Same padding means pad the input so that the output is the same size
    conv_1 = Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    conv_1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv_1)

    conv_2_1 = Conv2D(64, (3,3), padding='same', activation='relu')(conv_1)
    conv_2_1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv_2_1)

    conv_2_2 = Conv2D(64, (3,3), padding='same', activation='relu')(conv_1)
    conv_2_2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv_2_2)

    conv_3_1 = Conv2D(256, (3,3), padding='same', activation='relu')(conv_2_1)

    conv_3_2 = Conv2D(256, (3,3), padding='same', activation='relu')(conv_2_2)

    conv_3 = Concatenate()([conv_3_1, conv_3_2])

    # This layer wasn't clear from the specs, but I usually maxpool all my conv
    # layers.
    conv_3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv_3)

    # These both comprise one layer according to the specs.
    flatten = Flatten()(conv_3)
    fc_1 = Dense(1000, activation='relu')(flatten)

    fc_2 = Dense(500, activation='relu')(fc_1)

    output = Dense(units=10, activation='softmax')(fc_2)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=config.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

# Load MNIST data, reshape, and scale.
(x_train, y_train), (x_val, y_val) = mnist.load_data()
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
x_train = x_train/255
x_val = x_val/255

# Labels
y_train = np_utils.to_categorical(y_train, 10)
y_val = np_utils.to_categorical(y_val, 10)

# Create and fit the model; the model at the epoch with the best validation accuracy is saved.
model = create_model()
checkpointer = ModelCheckpoint(filepath='model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, period=1)
model.fit(x=x_train, y=y_train, batch_size=config.batch_size, epochs=config.epochs, callbacks=[checkpointer], validation_data=(x_val,y_val))
