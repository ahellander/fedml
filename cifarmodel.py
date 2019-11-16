from fedml import BaseLearner
import keras
from keras.utils import to_categorical
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

    # initiate Adam optimizer
    opt = keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)
    # opt = keras.optimizers.SGD(learning_rate=0.001)#, decay=1e-6)

    # Let's train the model using Adam
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
        # print("Before average weights -- virtual memory used: ", psutil.virtual_memory()[2], "%")

        weights = [model.model.get_weights() for model in models]

        avg_w = []
        avg_std = []
        for l in range(len(weights[0])):
            lay_l = np.array([w[l] for w in weights])
            # print("mean std layer ", l, ": ", np.mean(np.std(lay_l,0)))
            avg_std.append(np.mean(np.std(lay_l,0)))
            weight_l_avg = np.mean(lay_l,0)
            avg_w.append(weight_l_avg)
        mean_avg = np.mean(np.array(avg_std))
        # print("weights mean std: ", mean_avg)
        return avg_w, mean_avg

    def set_weights(self,weights):
        self.model.set_weights(weights)

    def predict(self, x):

        return to_categorical(self.model.predict_classes(x), num_classes=10)
        # return self.model.predict(x)

    def partial_fit(self, x, y, data_order, classes=None, data_set_index=0, training_steps=None):
        """ Do a partial fit. """
        # print("partial fit start -- virtual memory used: ", psutil.virtual_memory()[2], "%")
        batch_size = 32
        epochs = 1
        data_augmentation = True
        print("partial fit starts")
        if batch_size == "inf":
            batch_size = x.shape[0]

        if training_steps is not None:
            print("training steps: ", training_steps)
            epochs = 1
            start_ind = data_set_index
            end_ind = start_ind + batch_size * training_steps
            ind = []
            while end_ind > x.shape[0]:
                end_ind = end_ind - x.shape[0]
                ind += list(data_order[np.arange(start_ind, x.shape[0])])
                start_ind = 0
                data_order = np.random.permutation(x.shape[0])

            ind += list(data_order[np.arange(start_ind,end_ind)])
            data_set_index = end_ind

        else:
            print("training steps: ", training_steps)
            ind = np.arange(x.shape[0])
            print("ind: ", ind)

        if not data_augmentation:
            print('Not using data augmentation.')
            self.model.fit(x[ind], y[ind],
                      batch_size=batch_size,
                      epochs=epochs,
                      shuffle=False)
        else:
            # print('Using real-time data augmentation.')
            # print("before training(inside partial fit) -- virtual memory used: ", psutil.virtual_memory()[2], "%")

            # This will do preprocessing and realtime data augmentation:
            if self.datagen is None:
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
                self.datagen.fit(x)

            # Fit the model on the batches generated by datagen.flow().
            # print("before training(inside partial fit after datagen) -- virtual memory used: ", psutil.virtual_memory()[2], "%")
            self.model.fit_generator(self.datagen.flow(x[ind], y[ind],
                                                       batch_size=batch_size),
                                                       epochs=epochs,
                                                       workers=4,
                                                       shuffle=False)
            return data_set_index, data_order

