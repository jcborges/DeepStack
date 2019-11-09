"""
Module representing the Base-Learners, Members of an Ensemble
"""
import numpy as np
import os
from keras.models import load_model


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
        self.name = name
        self.train_probs = np.array(train_probs)
        self.train_classes = np.array(train_classes)
        self.val_probs = np.array(val_probs)
        self.val_classes = np.array(val_classes)
        self.submission_probs = np.array(submission_probs)

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
        name = folder.split(os.sep)[-1].replace(os.sep, "")
        if folder[-1] == os.sep:
            name = folder.split(os.sep)[-2].replace(os.sep, "")
        train_probs = np.load(os.path.join(folder, "train_probs.npy"))
        train_classes = np.load(os.path.join(folder, "train_classes.npy"))
        val_probs = np.load(os.path.join(folder, "val_probs.npy"))
        val_classes = np.load(os.path.join(folder, "val_classes.npy"))
        submission_probs = None
        if os.path.isfile(os.path.join(folder, "submission_probs.npy")):
            submission_probs = np.load(
                os.path.join(folder, "submission_probs.npy"))
        member = Member(name, train_probs, train_classes, val_probs,
                        val_classes, submission_probs)
        print("Loaded", name)
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
        np.save(folder + self.name + "/val_probs.npy", self.val_probs)
        np.save(folder + self.name + "/train_probs.npy", self.train_probs)
        np.save(folder + self.name + "/val_classes.npy", self.val_classes)
        np.save(folder + self.name + "/train_classes.npy", self.train_classes)
        if self.submission_probs is not None:
            np.save(folder + self.name + "/submission_probs.npy",
                    self.submission_probs)


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
            train_batches: training data for the Meta-Learner.
                Either a Keras `DataGenerator` or a tuple
                with training set (X, y).
            val_batches: validation data for the Meta-Learner.
                Either a Keras `DataGenerator` or a tuple
                with validation set (X, y).
            submission_probs: the final submission prediction probabilities
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
        if train_batches is not None:
            self._calculate_train_predictions(train_batches)
        if val_batches is not None:
            self._calculate_val_predictions(val_batches)

    def _test_datatuple(self, datatuple):
        assert(len(datatuple) == 2)
        assert(datatuple[0].shape[0] == datatuple[1].shape[0])

    def _calculate_predictions(self, batches):
        if hasattr(batches, 'shuffle'):
            batches.reset()
            batches.shuffle = False
        if type(batches) is tuple:
            self._test_datatuple(batches)
            return self.model.predict(batches[0])
        return self.model.predict_generator(
            batches, steps=(batches.n // batches.batch_size) + 1, verbose=1)

    def _calculate_val_predictions(self, val_batches):
        if type(val_batches) is tuple:
            self.val_classes = val_batches[1]
        elif hasattr(val_batches, 'classes'):
            self.val_classes = np.array(val_batches.classes)
        elif hasattr(val_batches, 'y'):
            self.val_classes = np.array(val_batches.y)
        else:
            raise ValueError("No known class in data batch")
        self.val_probs = self._calculate_predictions(val_batches)
        return self.val_probs

    def _calculate_train_predictions(self, train_batches):
        if type(train_batches) is tuple:
            self.train_classes = train_batches[1]
        elif hasattr(train_batches, 'classes'):
            self.train_classes = np.array(train_batches.classes)
        elif hasattr(train_batches, 'y'):
            self.train_classes = np.array(train_batches.y)
        else:
            raise ValueError("No known class in data batch")
        self.train_probs = self._calculate_predictions(train_batches)
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
