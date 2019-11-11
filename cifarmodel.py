from fedml import BaseLearner
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import psutil


def create_cifarmodel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


class KerasSequentialCifar(BaseLearner):
    """  Keras Sequential base learner."""

    def __init__(self):
        self.model = create_cifarmodel()
        self.datagen =None


    @staticmethod
    def average_weights(models):
        """ fdfdsfs """
        print("Before average weights -- virtual memory used: ", psutil.virtual_memory()[2], "%")

        weights = [model.model.get_weights() for model in models]

        avg_w = []
        for l in range(len(weights[0])):
            lay_l = np.array([w[l] for w in weights])
            print("mean std layer ", l, ": ", np.mean(np.std(lay_l,0)))
            weight_l_avg = np.mean(lay_l,0)
            avg_w.append(weight_l_avg)

        return avg_w

    def set_weights(self,weights):
        self.model.set_weights(weights)

    def predict(self, x):
        return self.model.predict(x)

    def partial_fit(self,x,y,classes=None):
        """ Do a partial fit. """
        print("partial fit start -- virtual memory used: ", psutil.virtual_memory()[2], "%")

        batch_size = 32
        epochs = 1
        save_as = 'cifarmodel'
        data_augmentation = True
        x_train = x
        y_train = y


        if not data_augmentation:
            print('Not using data augmentation.')
            self.model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      shuffle=True)
        else:
            print('Using real-time data augmentation.')
            print("before training(inside partial fit) -- virtual memory used: ", psutil.virtual_memory()[2], "%")

            # This will do preprocessing and realtime data augmentation:
            if self.datagen is None:
                print("initiate datagen")
                self.datagen = ImageDataGenerator(
                    featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    zca_epsilon=1e-06,  # epsilon for ZCA whitening
                    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                    # randomly shift images horizontally (fraction of total width)
                    width_shift_range=0.1,
                    # randomly shift images vertically (fraction of total height)
                    height_shift_range=0.1,
                    shear_range=0.,  # set range for random shear
                    zoom_range=0.,  # set range for random zoom
                    channel_shift_range=0.,  # set range for random channel shifts
                    # set mode for filling points outside the input boundaries
                    fill_mode='nearest',
                    cval=0.,  # value used for fill_mode = "constant"
                    horizontal_flip=True,  # randomly flip images
                    vertical_flip=False,  # randomly flip images
                    # set rescaling factor (applied before any other transformation)
                    rescale=None,
                    # set function that will be applied on each input
                    preprocessing_function=None,
                    # image data format, either "channels_first" or "channels_last"
                    data_format=None)

                # Compute quantities required for feature-wise normalization
                # (std, mean, and principal components if ZCA whitening is applied).
                self.datagen.fit(x_train)

            # Fit the model on the batches generated by datagen.flow().
            print("before training(inside partial fit after datagen) -- virtual memory used: ", psutil.virtual_memory()[2], "%")

            self.model.fit_generator(self.datagen.flow(x_train, y_train,
                                                       batch_size=batch_size),
                                          epochs=epochs,
                                          workers=4)
            print("after training(inside partial fit) -- virtual memory used: ", psutil.virtual_memory()[2], "%")


