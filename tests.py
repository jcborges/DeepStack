import pytest
from deepstack import Member
from deepstack import DirichletEnsemble
from sklearn.datasets.samples_generator import make_blobs
from keras.models import Sequential
from keras.layers import Dense
import random
import tensorflow as tf
import numpy as np

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