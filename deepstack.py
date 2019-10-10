import numpy as np
from sklearn import metrics
import warnings
from abc import abstractmethod
from sklearn.ensemble import RandomForestRegressor
import os
from keras.models import load_model
import joblib
import glob


class Member:
    """
    Representation of a single keras model member (Base-Learner) of an Ensemble
    """
    def __init__(self, name=None, keras_model=None, train_batches=None, val_batches=None,
                 submission_probs=None, keras_modelpath=None, keras_kwargs={}):
        """
        Constructor of a Keras Ensemble Member for a Binary Classification (Image Recognition) Task.
        Internal class probabilities are calculates based on ImageDataGenerators.
        If you wish to provide these directly, use `from_probs` constructor
        Args:
            name: name of the model. Must be unique.
            model: the (pre-trained) keras model. Or provide `keras_modelpath` instead.
            train_batches: an instance of Keras `ImageDataGenerator` for training the Meta-Learner
            val_batches: an instance of Keras `ImageDataGenerator` for validating the Meta-Learner
            submission_probs: the submission prediction probabilities
            keras_modelpath: path to load keras model from (if `model` argument is None)
            keras_kwargs: kwargs for keras `load_model` (if `model` argument is None)
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

    def __repr__(self):
        return "<Member: " + self.name + ">"

    def __str__(self):
        return "Member: " + self.name

    def __eq__(self, other):
        c1 = self.name == other.name
        c2 = _compare_arrays(self.train_classes, other.train_classes)
        c3 = _compare_arrays(self.val_classes, other.val_classes)
        c4 = _compare_arrays(self.val_probs, other.val_probs)
        c5 = _compare_arrays(self.train_probs, other.train_probs)
        return c1 and c2 and c3 and c4 and c5

    @classmethod
    def from_probs(cls, name, train_probs, train_classes, val_probs, val_classes, submission_probs):
        """
        Constructor based on class probabilities, not on `ImageDataGenerator`.
        Useful if one wants to calculated the keras model's probabilities idependently.
        Args:
            name: name of the model. Must be unique.
            train_probs: probabilities of positive class for training the Meta-Learner
            train_classes: ground truth (classes) for training the Meta-Learner
            val_probs: probabilities of positive class for validating the Meta-Learner
            val_classes: ground truth (classes) for validating the Meta-Learner
            submission_probs: the submission prediction probabilities
        Returns:
            a Member object
        """
        member = cls(name=name)
        member.train_probs = np.array(train_probs)
        member.train_classes = np.array(train_classes)
        member.val_probs = np.array(val_probs)
        member.val_classes = np.array(val_classes)
        member.submission_probs = np.array(submission_probs)
        return member

    @classmethod
    def load(cls, folder=None):
        """Loads base-learner from directory
        Args:
            folder: directory where member is saved
        Returns:
            loaded Member object
        """
        path = os.path.join(folder, "member.joblib")
        member = joblib.load(path)
        return member

    def _calculate_predictions(self, batches):  # TODO: call automatically for dirichlet ensemble
        if hasattr(batches, 'shuffle'):
            batches.reset()
            batches.shuffle = False
        preds = self.model.predict_generator(batches, steps=(batches.n // 32) + 1, verbose=1)
        if preds.shape[0] > 1:
            probs = preds[:, 0]
        else:
            probs = preds
        return probs

    def _calculate_val_predictions(self, val_batches):  # TODO: call automatically for dirichlet ensemble
        self.val_probs = self._calculate_predictions(val_batches)
        self.val_classes = np.array(val_batches.classes)
        return self.val_probs

    def _calculate_train_predictions(self, train_batches):  # TODO: call automatically for dirichlet ensemble
        self.train_probs = self._calculate_predictions(train_batches)
        self.train_classes = np.array(train_batches.classes)
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

    def save(self, folder="./premodels/", save_kerasmodel=False):
        """
        Saves member object to folder
        Args:
            folder: the folder where models should be saved to. Create if not exists.
            save_kerasmodel: if it should save the keras model as part of object.
                Recommendation is to load keras separately with method `load_kerasmodel`
        """
        if folder[-1] != os.sep:
            folder += os.sep
        if not os.path.exists(folder):
            os.mkdir(folder)
        if not os.path.exists(folder + self.name):
            os.mkdir(folder + self.name)
        if save_kerasmodel:
            joblib.dump(self, os.path.join(folder + self.name, "member.joblib"))
        else:
            temp = self.model
            self.model = None  # Remove Keras model from variable
            joblib.dump(self, os.path.join(folder + self.name, "member.joblib"))
            self.model = temp


class Ensemble(object):
    """Base Ensemble Definition."""

    @abstractmethod
    def add_member(self, member):
        """
        Adds a model Member to the Ensemble

        Args:
            member: the model Member
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self):
        """
        Fit method to provided ensemble members
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        """
        Predict using fitted ensemble members
        """
        raise NotImplementedError


class DirichletEnsemble(Ensemble):
    """
    Representation of an ensemble of Keras Models for a Binary Classification Task.
    It weights the ensemble members optimizing the AUC-Score based on a validation dataset.
    The weight optimization search is performed with randomized search based on the dirichlet distribution.
    """
    def __init__(self, val_classes, positive_class=True, N=10000):
        """
        Constructor of a Keras Ensemble Binary Classifier
        Args:
            val_classes: classes of validation dataset, from which the enseble weights will be calculated
            N: the number of times weights should be (randomly) tried out, sampled from a dirichlet distribution
            positive_class:  specifies which class from `val_classes` should count as positive for the AUC calculation
        """
        self.val_classes = val_classes
        self.n = N
        self.positive_class = positive_class
        # Initialize Parameters:
        self.members = []
        self.bestweights = []
        self.probabilities = None
        self._nmembers = 0
        self.bestauc = 0
        self.fitted = False

    def add_member(self, member):
        """
        Adds a ensemble Member to the Stack
        Args:
            member: an instance of class `Member`
        """
        self.members.append(member)
        self._nmembers += 1

    def fit(self, verbose=False):
        """
        Calculates ensemble weights, optimizing the AUC Binary Classification Metric using randomized
        search with the dirichlet distribution.
        """
        assert(self.val_classes is not None)
        assert(len(self.members) > 1)

        aucbest = 0
        rsbest = None
        for i in range(self.n):
            rs = np.random.dirichlet(np.ones(self._nmembers), size=1)[0]
            preds = np.sum(np.array([self.members[i].val_probs * rs[i] for i in range(self._nmembers)]), axis=0)
            auc = metrics.roc_auc_score([x == self.positive_class for x in self.val_classes], preds)
            if auc > aucbest:
                if verbose:
                    print(auc, i, rs)  # TODO: Proper logging
                aucbest = auc
                rsbest = rs
        self.bestweights = rsbest
        self.bestauc = aucbest

    def predict(self):
        """
        Returns the weighed probabilities of the ensemble members
        Returns:
            the predicted probabilities as np.array
        """
        self.probabilities = np.sum(np.array([self.bestweights[i] * self.members[i].submission_probs
                                              for i in range(self._nmembers)]), axis=0)
        return self.probabilities

    def describe(self):
        """
        Prints information about the ensemble members and its weights as well as single and ensemble AUC performance
        on validation dataset.
        """
        modelbestauc = 0
        for i in range(self._nmembers):
            model = self.members[i]
            auc = metrics.roc_auc_score([x == self.positive_class for x in self.val_classes], model.val_probs)
            if auc > modelbestauc:
                modelbestauc = auc
            print(self.members[i].name, "- Weight:", self.bestweights[i], "- Single AUC:", auc)
        print("DirichletEnsemble AUC:", modelbestauc)
        return


class StackEnsemble(Ensemble):
    def __init__(self, model=None):
        """
        Constructor of a Stacking Ensemble, with Keras Models as Base-Learners.
        It supports by now only keras binary classifiers. It constructs a meta-learner that predicts
        the probability of the positive class as a regression problem. 
        Args:
            model: ensemble model which should serve as meta-model. Sklearn RandomForestRegressor per default.
        """
        if model is None:
            self.model = RandomForestRegressor(verbose=1, n_estimators=100, max_depth=3)
        else:
            self.model = model
        # Initialize Parameters:
        self.members = []
        self._nmembers = 0
        self.predictions = None

    def __repr__(self):
        reps = [member.name for member in self.members]
        return "<StackEnsemble: [" + ", ".join(reps) + "]>"

    def __str__(self):
        reps = [member.name for member in self.members]
        return "StackEnsemble: with" + str(self._nmembers) + " Base-Learners [" + ", ".join(reps) + "]"

    def add_member(self, member):
        """
        Adds a ensemble Member to the Stack
        Args:
            member: an instance of class `Member`
        """
        self.members.append(member)
        self._nmembers += 1
        if member.val_probs is None:
            try:
                member.val_probs = member._calculate_val_predictions()
            except Exception as e:
                warnings.warn(str(e))
        if member.train_probs is None:
            try:
                member.train_probs = member._calculate_train_predictions()
            except Exception as e:
                warnings.warn(str(e))

    def fit(self, X=None, y=None, kwargs={}):
        """
        Trains the meta-model
        Args:
            X: training data for meta-learner
            y: training classes for meta-learner
            kwargs: further arguments for the fit function
        """
        assert(len(self.members) > 1)
        # Assumption: all members have same train_batches.classes
        if X is None or y is None:
            return self._fit_train()
        return self.model.fit(X, y, **kwargs)

    def predict(self, X=None, kwargs={}):
        """
        Meta-Model prediction for the probabilities of the positive class as a regression problem
        Args:
            X: input data to be predicted
            kwargs: further arguments for prediction function
        Returns:
            the predicted probabilities as np.array
        """
        if X is None:
            X = self._get_pred_X()
        try:
            self.predictions = self.model.predict_proba(X, **kwargs)[:, 0]
        except Exception as e:
            print(e)
            self.predictions = self.model.predict(X, **kwargs)
        return self.predictions

    def describe(self, probabilities_val=None):
        """
        Prints information about the performance of base and meta learners based on validation data
        Args:
            probabilities_val: (optional) probabilities/prediction on validation data
        """
        modelbestauc = 0
        if probabilities_val is None:
            probabilities_val = self._predict_val()
        val_classes = self.members[0].val_classes  # Assumption: all members have same val_classes
        for i in range(self._nmembers):
            member = self.members[i]
            valprobs = member.val_probs
            auc = metrics.roc_auc_score(member.val_classes, valprobs)
            if auc < 0.5:
                valprobs = [1 - x for x in valprobs]
            auc = metrics.roc_auc_score(member.val_classes, valprobs)
            if auc > modelbestauc:
                modelbestauc = auc
            print(member.name, "AUC:", auc)
        auc = metrics.roc_auc_score(val_classes, probabilities_val)
        if auc < 0.5:
            probabilities_val = [1 - x for x in probabilities_val]
        auc = metrics.roc_auc_score(val_classes, probabilities_val)
        print("Ensemble AUC:", auc)
        return auc

    def _get_X(self, attrname):
        X = []
        probs = getattr(self.members[0], attrname)
        for i in range(len(probs)):  # Assumption: all members have same train_probs length
            preds = []
            for member in self.members:
                preds.append(getattr(member, attrname)[i])
            X.append(preds)
        return np.array(X)

    def _get_train_X(self):
        return self._get_X("train_probs")

    def _get_val_X(self):
        return self._get_X("val_probs")

    def _get_pred_X(self):
        return self._get_X("submission_probs")

    def _fit_train(self):
        return self.fit(self._get_train_X(), self.members[0].train_classes)

    def _fit_submission(self):
        """
        Fits model on training and validation data.
        Useful when training the meta-learner for final submission prediction
        """
        X1 = self._get_train_X()
        X2 = self._get_val_X()
        y1 = self.members[0].train_classes
        y2 = self.members[0].val_classes
        X = np.concatenate((X1, X2))
        y = np.concatenate((y1, y2))
        return self.fit(X, y)

    def _predict_val(self):
        return self.predict(self._get_val_X())

    def _test(self):
        """
        Test assumption that all members' classes have same shape and values. Names should be unique.
        This is an internal condition for class structures.
        """
        if self._nmembers < 2:
            return True
        t1 = [(_compare_arrays(self.members[i].train_classes,
                               self.members[i + 1].train_classes)) for i in range(self._nmembers - 1)]
        t2 = [(_compare_arrays(self.members[i].val_classes,
                               self.members[i + 1].val_classes)) for i in range(self._nmembers - 1)]
        assert(np.sum(t1) == self._nmembers - 1)
        assert(np.sum(t2) == self._nmembers - 1)
        names = [self.members[i].name for i in range(self._nmembers)]
        assert(len(list(names)) == len(set(names)))
        return True

    def save(self, folder="./premodels/", save_kerasmodel=False):  # TODO: Document
        """
        Saves meta-learner and base-learner of ensemble into folder / directory
        Args:
            folder: the folder where models should be saved to. Create if not exists.
            save_kerasmodel: if members / base-learners should save the keras model as part of object.
        """
        if not os.path.exists(folder):
            os.mkdir(folder)
        [member.save(folder=folder, save_kerasmodel=save_kerasmodel) for member in self.members]
        temp = self.members
        self.members = None  # Reset base-learners. These are loaded idependently
        self._nmembers = 0
        joblib.dump(self, os.path.join(folder, "stackensemble.joblib"))
        self.members = temp
        self._nmembers = len(self.members)
        return self

    @classmethod
    def load(cls, folder="./premodels/"):
        """
        Loads meta-learner and base-learners from folder / directory
        Args:
            folder: directory where models should be loaded from
        Returns:
            loaded StackEnsemble with Members
        """
        stack = joblib.load(os.path.join(folder, "stackensemble.joblib"))
        stack.members = []
        if folder[-1] != os.sep:
            folder += os.sep
        for fn in glob.glob(folder + "**/"):
            member = Member.load(fn)
            stack.add_member(member)
        return stack


def _compare_arrays(a1, a2):
    if a1 is None and a2 is None:
        return True
    if a1 is None and a2 is not None:
        return False
    if a1 is not None and a2 is None:
        return False
    c1 = a1.shape == a2.shape
    c2 = np.sum(a1 == a2) == len(a1)
    return c1 and c2
