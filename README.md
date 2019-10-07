# DeepStack

DeepStack: Ensembling Keras Deep Learning Models into the next Performance Level

[![Travis](https://travis-ci.com/jcborges/DeepStack.svg?branch=master)](https://travis-ci.com/jcborges/DeepStack)
---

DeepStack is a Python module for building Deep Learning Ensembles built on top of Keras and distributed under the MIT license.

DeepStack was started in 2019 by Julio Borges out of a Deep Learning Competition. 
It is still under development and currently supports binary classifiers.
DeepStack is currently looking for a team of volunteers and welcomes new contributors of all experience levels. 


## Installation
```
pip install git+https://github.com/jcborges/DeepStack
```

## Stacking
Stacking is based on training a (Sklearn) Meta-Learner on top of Keras pre-trained Base-Learners.
DeepStack offer an interface to fit the Meta-Learner on the predictions of the Base-Learners.

A running example can be found on `tests.py`

#### Usage 

```python
from deepstack import Member
from deepstack import StackEnsemble

model1 = ...  # A Keras Model
train_batches1 =  ...  # A Keras Data Iterator - Training Data for Meta-Learner
val_batches1 =  ...  # A Keras Data Iterator - Validation Data for Meta-Learner
pred_batches1 =  ...  # A Keras Data Iterator - Data to be predicted (no classes necessary)
member1 = Member(name="Model1", keras_model=model1, train_batches=train_batches1, val_batches=val_batches1, pred_batches = pred_batches1)

model2 = ...  # A Keras Model
train_batches2 =  ...  # A Keras Data Iterator - Training Data for Meta-Learner
val_batches2 =  ...  # A Keras Data Iterator - Validation Data for Meta-Learner
pred_batches2 =  ...  # A Keras Data Iterator - Data to be predicted (no classes necessary)
member2 = Member(name="Model2", keras_model=model2, train_batches=train_batches2, val_batches=val_batches2, pred_batches = pred_batches2)

stack = StackEnsemble()
stack.add_member(member1)  # Assumption: the data iterators of base-learners iterate over the same data and have same shape and classes.
stack.add_member(member2)
stack.fit()  # Fits meta-learner based on training batches from its members (base-learners)
stack.describe()  # Prints information about ensemble performance based on validation data
prediction = stack.predict()
```

## Randomized Weighted Ensemble
Ensemble Technique that weights the prediction of each ensemble member, combining the weights to calculate a combined prediction.  Weight optimization search is performed with randomized search based on the dirichlet distribution on a validation dataset. 

```python
from deepstack import Member
from deepstack import DirichletEnsemble
from sklearn.datasets.samples_generator import make_blobs
from keras.models import Sequential
from keras.layers import Dense
import random
import tensorflow as tf
import numpy as np

s=3
np.random.seed(seed=s)
tf.set_random_seed(seed=s)

def fit_model(trainX, trainy):
    model = Sequential()
    model.add(Dense(random.randint(20,30), input_dim=2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(trainX, trainy, epochs=50, verbose=0)
    return model

X, y = make_blobs(n_samples=1100, centers=2, n_features=2, cluster_std=4, random_state=s)
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
n_members = 5
stack = DirichletEnsemble(testy, N=5000, positive_class=1)
for i in range(n_members):
    model = fit_model(trainX, trainy)
    probs = model.predict(testX, verbose=0)
    m = Member(model, name="Model " + str(i))
    m.val_probs = probs
    stack.add_member(m)
stack.fit()
stack.describe()
```

Output:

```
Model 0 - Weight: 0.13174960351890574 - Single AUC: 0.8866956841785216
Model 1 - Weight: 0.3640771003076258 - Single AUC: 0.8762141748411771
Model 2 - Weight: 0.019055656117963362 - Single AUC: 0.8820790193787905
Model 3 - Weight: 0.4610742287201483 - Single AUC: 0.8850194427997632
Model 4 - Weight: 0.02404341133535681 - Single AUC: 0.8855835240274599

Better Performance. Achieved an AUC of 0.8898001312188955 and the best ensemble member alone achieves an AUC of 0.8866956841785216
```


## Citing DeepStack
If you use DeepStack in a scientific publication, we would appreciate citations:

```bibtex
@misc{
    title   = {DeepStack: Ensembling Keras Deep Learning Models into the next Performance Level},
    author  = {Julio Borges},
    url = {https://github.com/jcborges/DeepStack},
    date = {2019}
}
```
 
