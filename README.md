# DeepStack

DeepStack: Ensembling Keras Deep Learning Models into the next Performance Level

[![Travis](https://travis-ci.com/jcborges/DeepStack.svg?branch=master)](https://travis-ci.com/jcborges/DeepStack)
---

DeepStack is a Python module for building Deep Learning Ensembles built on top of Keras and distributed under the MIT license.
DeepStack is still under development and currently supports binary classifiers. 


## Installation
```
pip install git+https://github.com/jcborges/DeepStack
```

## Stacking
Stacking is based on training a (Sklearn) Meta-Learner on top of Keras pre-trained Base-Learners.
DeepStack offer an interface to fit the Meta-Learner on the predictions of the Base-Learners.


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

It follows the same interface of the StackEnsemble. A running example can be found in `tests.py`.
