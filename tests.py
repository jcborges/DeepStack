import pytest
from deepstack import Member
from deepstack import DirichletEnsemble
from deepstack import StackEnsemble
from sklearn.datasets.samples_generator import make_blobs
from keras.models import Sequential
from keras.layers import Dense
import random
import tensorflow as tf
import numpy as np
import wget 
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import LabelEncoder

def _fit_dirichlet_model(trainX, trainy):
        model = Sequential()
        model.add(Dense(random.randint(20,30), input_dim=2, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(trainX, trainy, epochs=50, verbose=0)
        return model

def test_dirichletensemble():
    """
    Tests if builsing an dirichlet ensemble is running without problems
    """
    np.random.seed(seed=2)
    X, y = make_blobs(n_samples=1100, centers=2, n_features=2, cluster_std=4, random_state=2)
    n_train = 100
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    n_members = 5
    stack = DirichletEnsemble(testy, N=7000, positive_class=1)
    for i in range(n_members):
        model = _fit_dirichlet_model(trainX, trainy)
        probs = model.predict(testX, verbose=0)
        m = Member(model, name="Model " + str(i))
        m.val_probs = probs
        stack.add_member(m)
    stack.fit()
    stack.describe()
    return True

class CustomIterator:
    def __init__(self, X_data, y_data, batch_size):
        self.X_data, self.y_data, self.batch_size = X_data, y_data, batch_size
        self.n = len(X_data)

        samples_per_epoch = self.X_data.shape[0]
        self.number_of_batches = samples_per_epoch/self.batch_size
        self.counter = 0
        self.shuffle = False

        encoder = LabelEncoder()
        encoder.fit(y_data)
        self.y_data = encoder.transform(y_data)
        self.classes = y_data

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter >= self.number_of_batches:
            self.counter=0
        X_batch = np.array(self.X_data[self.batch_size*self.counter:self.batch_size*(self.counter+1)])
        y_batch = np.array(self.y_data[self.batch_size*self.counter:self.batch_size*(self.counter+1)])
        self.counter += 1
        out = np.array([[0,1] if x == 1 else [1,0] for x in y_batch])
        return X_batch, out
    
    def reset(self):
        self.counter = 0

def test_stackensemble():
    if not os.path.isfile("sonar.csv"):
        wget.download("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data", "sonar.csv")
    
    df = pd.read_csv("sonar.csv", header=None)
    classes = df[60].map({"M":0, "R":1})
    del df[60]
    X_train, X_test, y_train, y_test = train_test_split(df, classes, test_size=2/3, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=1)

    model1 = Sequential()
    model1.add(Dense(60, input_dim=60, activation='relu'))
    model1.add(Dense(30, activation='relu'))
    model1.add(Dense(2, activation='softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    train_batches1 = CustomIterator(X_train, y_train, 32)
    val_batches1 = CustomIterator(X_val, y_val, 32)
    test_batches1 = CustomIterator(X_test, y_test, 32)
    model1.fit_generator(
    train_batches1,
    validation_data=val_batches1,
    steps_per_epoch=train_batches1.n//32,
    validation_steps=val_batches1.n//32,
    epochs=30
    )
    member1=Member(model1, train_batches=val_batches1, val_batches=test_batches1, name="Model1")

    model2 = Sequential()
    model2.add(Dense(15, input_dim=60, activation='relu'))
    model2.add(Dense(7, activation='relu'))
    model2.add(Dense(2, activation='softmax'))
    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    train_batches2 = CustomIterator(X_train, y_train, 32)
    val_batches2 = CustomIterator(X_val, y_val, 32)
    test_batches2 = CustomIterator(X_test, y_test, 32)
    model2.fit_generator(
    train_batches2,
    validation_data=val_batches2,
    steps_per_epoch=train_batches2.n//32,
    validation_steps=val_batches2.n//32,
    epochs=30
    )
    member2=Member(model2, train_batches=val_batches2, val_batches=test_batches2, name="Model2")

    stack = StackEnsemble()
    stack.add_member(member1)
    stack.add_member(member2)
    stack._test()
    stack.fit()
    stack.describe()    