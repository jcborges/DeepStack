import sys
sys.path.append(".")
import os
os.environ['KERAS_BACKEND'] = "tensorflow"
from deepstack.base import KerasMember
from deepstack.ensemble import DirichletEnsemble
from deepstack.ensemble import StackEnsemble
from sklearn.datasets.samples_generator import make_blobs
from keras.models import Sequential
import random
import numpy as np
import wget 
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sys import platform
from keras.utils import to_categorical
from keras.datasets import cifar10
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from sklearn.ensemble import RandomForestRegressor
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import BatchNormalization


def _get_fitted_random_model(trainX, trainy):
    model = Sequential()
    model.add(Dense(random.randint(20, 30), input_dim=2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
    model.fit(trainX, trainy, epochs=50, verbose=0)
    return model


def test_dirichletensemble():
    """
    Tests if builsing an dirichlet ensemble is running without problems
    """
    np.random.seed(seed=2)
    X, y = make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=4, 
                      random_state=2)
    n_train = 100
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    n_members = 5
    stack = DirichletEnsemble(N=5000)
    for i in range(n_members):
        model = _get_fitted_random_model(trainX, trainy)
        train_batches = CustomIterator(trainX, trainy, 32)
        val_batches = CustomIterator(testX, testy, 32)
        m = KerasMember(keras_model=model, name="Model " + str(i),
                        train_batches=train_batches, val_batches=val_batches)
        stack.add_member(m)
    stack.fit()
    stack.describe()
    return True


class CustomIterator:
    def __init__(self, X_data, y_data, batch_size):
        self.X_data, self.y_data, self.batch_size = X_data, y_data, batch_size
        self.n = len(X_data)
        samples_per_epoch = self.X_data.shape[0]
        self.number_of_batches = samples_per_epoch / self.batch_size
        self.counter = 0
        self.shuffle = False

        self.y_data = y_data
        self.classes = y_data
        self.y = y_data

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter >= self.number_of_batches:
            self.counter = 0
        X_batch = np.array(self.X_data[self.batch_size * self.counter:self.batch_size * (self.counter + 1)])
        y_batch = np.array(self.y_data[self.batch_size * self.counter:self.batch_size * (self.counter + 1)])
        self.counter += 1
        return X_batch, y_batch

    def reset(self):
        self.counter = 0


def test_stackensemble():
    if not os.path.isfile("sonar.csv"):
        wget.download("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data", "sonar.csv")

    df = pd.read_csv("sonar.csv", header=None)
    classes = df[60].map({"M": 0, "R": 1})
    del df[60]
    X_train, X_test, y_train, y_test = train_test_split(
        df, classes, test_size=2 / 3, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.5, random_state=1)

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)

    model1 = Sequential()
    model1.add(Dense(60, input_dim=60, activation='relu'))
    model1.add(Dense(30, activation='relu'))
    model1.add(Dense(2, activation='softmax'))
    model1.compile(loss='categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])
    train_batches1 = CustomIterator(X_train, y_train, 32)
    val_batches1 = CustomIterator(X_val, y_val, 32)
    test_batches1 = CustomIterator(X_test, y_test, 32)
    model1.fit_generator(
        train_batches1,
        validation_data=val_batches1,
        steps_per_epoch=train_batches1.n // 32,
        validation_steps=val_batches1.n // 32,
        epochs=10
    )
    member1 = KerasMember(keras_model=model1, train_batches=val_batches1,
                          val_batches=test_batches1, name="Model1")

    model2 = Sequential()
    model2.add(Dense(15, input_dim=60, activation='relu'))
    model2.add(Dense(7, activation='relu'))
    model2.add(Dense(2, activation='softmax'))
    model2.compile(loss='categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])
    train_batches2 = CustomIterator(X_train, y_train, 32)
    val_batches2 = CustomIterator(X_val, y_val, 32)
    test_batches2 = CustomIterator(X_test, y_test, 32)
    model2.fit_generator(
        train_batches2,
        validation_data=val_batches2,
        steps_per_epoch=train_batches2.n // 32,
        validation_steps=val_batches2.n // 32,
        epochs=10
    )
    member2 = KerasMember(keras_model=model2, train_batches=val_batches2,
                          val_batches=test_batches2, name="Model2")

    if not os.path.exists("./premodels/"):
        os.mkdir("./premodels/")

    model1.save("./premodels/model1.h5")
    model2.save("./premodels/model2.h5")

    stack = StackEnsemble()
    stack.add_member(member1)
    stack.add_member(member2)
    stack._test()
    stack.fit()
    auc1 = stack.describe()
    stack.save()
    stack2 = StackEnsemble.load()
    auc2 = stack2.describe()
    stack2._test()

    if platform == "darwin":
        print(auc1, auc2)
        assert(auc1 == auc2)  # TODO: not working under linux

    member1.load_kerasmodel("./premodels/model1.h5")
    member2.load_kerasmodel("./premodels/model2.h5")
    stack3 = StackEnsemble()
    stack3.add_member(member1)
    stack3.add_member(member2)
    stack3.fit()
    auc3 = stack3.describe()
    stack3._test()
    stack3.save()

    stack4 = StackEnsemble.load()
    auc4 = stack4.describe()
    stack4._test()

    if platform == "darwin":
        print(auc3, auc4)
        assert(auc3 == auc4)   # TODO: not working under linux


def _load_cifar_dataset(trainsample=5000, testsample=500):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    trainindex = random.sample(list(range(x_train.shape[0])), trainsample)
    return x_train[trainindex, :, :, :], y_train[trainindex, :], x_test[0:testsample, :, :, :], y_test[0:testsample, :], x_test[testsample:testsample * 2, :, :, :], y_test[testsample:testsample * 2, :]


def _create_random_cifar_model(input_shape):
    weight_decay = 1e-4
    num_classes = 10
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=input_shape))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(random.randint(16, 64), (3, 3), padding='same', 
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(random.randint(0, 5) * 0.1))

    model.add(Conv2D(random.randint(16, 64), (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(random.randint(16, 64), (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(random.randint(0, 5) * 0.1))

    model.add(Conv2D(random.randint(64, 128), (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(random.randint(64, 128), (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(random.randint(0, 5) * 0.1))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    return model


def _get_random_cifar_model(batch_size=32):
    opt_rms = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
    datagen = ImageDataGenerator(rotation_range=90,
                                 width_shift_range=0.1, height_shift_range=0.1,
                                 horizontal_flip=True)
    x_train, y_train, x_val, y_val, x_test, y_test = _load_cifar_dataset()
    model = _create_random_cifar_model(input_shape=x_train.shape[1:])
    datagen.fit(x_train)
    model.compile(loss='categorical_crossentropy', optimizer=opt_rms,
                  metrics=['accuracy'])
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=1, verbose=1, validation_data=(x_test, y_test))

    tb = datagen.flow(x_val, y_val, batch_size=batch_size)
    vb = datagen.flow(x_test, y_test, batch_size=batch_size)
    return model, tb, vb


def test_cifar10():
    model1, tb1, vb1 = _get_random_cifar_model()
    model2, tb2, vb2 = _get_random_cifar_model()
    model3, tb3, vb3 = _get_random_cifar_model()

    member1 = KerasMember(name="model1", keras_model=model1,
                          train_batches=tb1, val_batches=vb1)

    member2 = KerasMember(name="model2", keras_model=model2,
                          train_batches=tb2, val_batches=vb2)

    member3 = KerasMember(name="model3", keras_model=model3,
                          train_batches=tb3, val_batches=vb3)

    stack = StackEnsemble()
    stack.model = RandomForestRegressor(verbose=1, n_estimators=3000,
                                        max_depth=3, n_jobs=4)
    stack.add_member(member1)
    stack.add_member(member2)
    stack.add_member(member3)
    stack.fit()
    stack.describe()
    stack._test()

    dstack = DirichletEnsemble(N=100)
    dstack.add_member(member1)
    dstack.add_member(member2)
    dstack.add_member(member3)
    dstack.fit()
    dstack.describe()
