"""
Module representing the Base-Learners, Members of an Ensemble
"""
import numpy as np
import os
from keras.models import load_model
import joblib


class Member:
    def __init__(self, name, train_probs, train_classes, val_probs,
                 val_classes, submission_probs):
        """
        Constructor for an Ensemble Member (Base-Learner) based on class'
        probabilities.
        Args:
            name: name of the member. Must be unique.
            train_probs: class' probabilities for training the Meta-Learner
            train_classes: ground truth (classes) for training the Meta-Learner
            val_probs: class' probabilitiess for validating the Meta-Learner
            val_classes: ground truth (classes) for validating the Meta-Learner
            submission_probs: the final (submission) prediction probabilities
        Returns:
            a Member object
        """
        self.train_probs = np.array(train_probs)
        self.train_classes = np.array(train_classes)
        self.val_probs = np.array(val_probs)
        self.val_classes = np.array(val_classes)
        self.submission_probs = np.array(submission_probs)
        return self

    def __repr__(self):
        return "<Member: " + self.name + ">"

    def __str__(self):
        return "Member: " + self.name

    def __eq__(self, other):
        c1 = self.name == other.name
        c2 = np.array_equal(self.train_classes, other.train_classes)
        c3 = np.array_equal(self.val_classes, other.val_classes)
        c4 = np.array_equal(self.val_probs, other.val_probs)
        c5 = np.array_equal(self.train_probs, other.train_probs)
        return c1 and c2 and c3 and c4 and c5

    @classmethod
    def load(cls, folder=None):
        """
        Loads base-learner from directory
        Args:
            folder: directory where member is saved
        Returns:
            loaded Member object
        """
        path = os.path.join(folder, "member.joblib")
        member = joblib.load(path)
        return member

    def save(self, folder="./premodels/"):
        """
        Saves member object to folder.
        Args:
            folder: the folder where models should be saved to.
            Create if not exists.
        """
        if folder[-1] != os.sep:
            folder += os.sep
        if not os.path.exists(folder):
            os.mkdir(folder)
        if not os.path.exists(folder + self.name):
            os.mkdir(folder + self.name)
        joblib.dump(self, os.path.join(folder + self.name, "member.joblib"))


class KerasMember(Member):
    """
    Representation of a single keras model member (Base-Learner) of an Ensemble
    """
    def __init__(self, name=None, keras_model=None, train_batches=None,
                 val_batches=None, submission_probs=None, keras_modelpath=None,
                 keras_kwargs={}):
        """
        Constructor of a Keras Ensemble Member.
        Internal class' probabilities are calculates based on DataGenerators.
        Args:
            name: name of the model. Must be unique.
            model: the (pre-trained) keras model.
                Or provide `keras_modelpath` instead.
            train_batches: an instance of Keras `DataGenerator` for training
                the Meta-Learner
            val_batches: an instance of Keras `DataGenerator` for validating
                the Meta-Learner
            submission_probs: the submission prediction probabilities
            keras_modelpath: path to load keras model from (if `model`
                argument is None)
            keras_kwargs: kwargs for keras `load_model` (if `model` argument
                is None)
        """
        assert(name is not None)
        self.name = name
        self.model = keras_model
        self.submission_probs = submission_probs
        # Initialize Params
        self.val_probs = None
        self.train_probs = None
        self.val_classes = None
        self.train_classes = None
        if (keras_model is None) and (keras_modelpath is not None):
            self.load_kerasmodel(self.keras_modelpath, self.keras_kwargs)
        if val_batches is not None:
            self._calculate_val_predictions(val_batches)
        if train_batches is not None:
            self._calculate_train_predictions(train_batches)

    def _calculate_predictions(self, batches):
        if hasattr(batches, 'shuffle'):
            batches.reset()
            batches.shuffle = False
        return self.model.predict_generator(
            batches, steps=(batches.n // batches.batch_size) + 1, verbose=1)

    def _calculate_val_predictions(self, val_batches):
        self.val_probs = self._calculate_predictions(val_batches)
        if hasattr(val_batches, 'classes'):
            self.val_classes = np.array(val_batches.classes)
        elif hasattr(val_batches, 'y'):
            self.val_classes = np.array(val_batches.y)
        else:
            raise("No known class in data batch")
        return self.val_probs

    def _calculate_train_predictions(self, train_batches):
        self.train_probs = self._calculate_predictions(train_batches)
        if hasattr(train_batches, 'classes'):
            self.train_classes = np.array(train_batches.classes)
        elif hasattr(train_batches, 'y'):
            self.train_classes = np.array(train_batches.y)
        else:
            raise("No known class in data batch")
        return self.train_probs

    def load_kerasmodel(self, keras_modelpath=None, keras_kwargs={}):
        """
        Utility method for loading Keras model
        Args:
            keras_modelpath: path to keras model
            keras_kwargs: arguments for keras `load_model`
        """
        if keras_kwargs is None:
            keras_kwargs = {}
        self.model = load_model(keras_modelpath, **keras_kwargs)
        print("Keras Model Loaded:", keras_modelpath)
        return self.model
