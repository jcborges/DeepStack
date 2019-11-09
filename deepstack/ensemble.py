"""
Module representing the Meta-Learners, containing an Ensemble of Base-Learners
"""
import numpy as np
from sklearn import metrics
import warnings
from abc import abstractmethod
from sklearn.ensemble import RandomForestRegressor
import os
import joblib
import glob
from deepstack.base import Member
from keras.utils import to_categorical


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
    Representation of an Dirichlet Ensemble
    It weights the ensemble members optimizing a Metric/Score based on a
    validation dataset. The weight optimization search is performed with
    randomized search based on the dirichlet distribution.
    """

    def __init__(self, N=10000, evaluation_metric=None, maximize=True):
        """
        Constructor of a Dirichlet Weighted Ensemble
        Args:
            N: the number of times weights should be (randomly) tried out,
                sampled from a dirichlet distribution
            evaluation_metric: (optional) evaluation metric function.
                Default: `sklearn.metrics.roc_auc_score`
            maximize: if evaluation metric should be maximized (otherwise minimized)
        """
        self.n = N
        self.evaluation_metric = evaluation_metric
        if evaluation_metric is None:
            self.evaluation_metric = metrics.roc_auc_score
        self.maximize = maximize
        # Initialize Parameters:
        self.members = []
        self.bestweights = []
        self.probabilities = None
        self._nmembers = 0
        self.bestscore = float("-inf") if maximize else float("inf")
        self.fitted = False

    def add_members(self, members):
        """
        Adds Members to the Ensemble
        Args:
            members: a list containing instances of class `Member`
        """
        for member in members:
            self.add_member(member)

    def add_member(self, member):
        """
        Adds a Member (Base-Learner) to the Ensemble
        Args:
            member: an instance of class `Member`
        """
        self.members.append(member)
        self._nmembers += 1

    def fit(self, verbose=False):
        """
        Calculates ensemble weights, optimizing the AUC Binary Classification
        Metric using randomized search with the dirichlet distribution.
        """
        assert(len(self.members) > 1)
        val_classes = self.members[0].val_classes

        best_ensemble_score = float("-inf") if self.maximize else float("inf")
        rsbest = None
        for i in range(self.n):
            rs = np.random.dirichlet(np.ones(self._nmembers), size=1)[0]
            preds = np.sum(np.array([self.members[i].val_probs * rs[i]
                                     for i in range(self._nmembers)]), axis=0)
            ensemble_score = _calculate_metric(val_classes, preds, self.evaluation_metric)
            max_flag = self.maximize and ensemble_score > best_ensemble_score
            min_flag = not(self.maximize) and ensemble_score < best_ensemble_score
            if max_flag or min_flag:
                if verbose:
                    print(ensemble_score, i, rs)  # TODO: Proper logging
                best_ensemble_score = ensemble_score
                rsbest = rs
        self.bestweights = rsbest
        self.bestscore = best_ensemble_score

    def predict(self):
        """
        Returns the weighed probabilities of the ensemble members
        Returns:
            the predicted probabilities as np.array
        """
        self.probabilities = np.sum(np.array([self.bestweights[i] *
                                              self.members[i].submission_probs
                                              for i in range(self._nmembers)]),
                                    axis=0)
        return self.probabilities

    def describe(self):
        """
        Prints information about the ensemble members and its weights as well
        as single and ensemble AUC performance on validation dataset.
        """
        for i in range(self._nmembers):
            member = self.members[i]
            model_score = _calculate_metric(member.val_classes,
                                            member.val_probs,
                                            metric=self.evaluation_metric)
            text = self.members[i].name + \
                " - Weight: {:1.4f} - {}: {:1.4f}".format(
                    self.bestweights[i],
                    self.evaluation_metric.__name__,
                    model_score)
            print(text)
        print("DirichletEnsemble {}: {:1.4f}".format(
            self.evaluation_metric.__name__,
            self.bestscore))
        return


class StackEnsemble(Ensemble):
    def __init__(self, model=None):
        """
        Constructor of a Stacking Ensemble.
        Args:
            model: ensemble model which should serve as meta-model.
                `sklearn.ensemble.RandomForestRegressor` per default for prediction class probabilities.
        """
        self.model = model
        if model is None:
            self.model = RandomForestRegressor(n_estimators=100, max_depth=3, n_jobs=20)
        # Initialize Parameters:
        self.members = []
        self._nmembers = 0
        self.predictions = None

    def __repr__(self):
        reps = [member.name for member in self.members]
        return "<StackEnsemble: [" + ", ".join(reps) + "]>"

    def __str__(self):
        reps = [member.name for member in self.members]
        return "StackEnsemble: with" + \
            str(self._nmembers) + " Base-Learners [" + ", ".join(reps) + "]"

    def add_members(self, members):
        """
        Adds ensemble Members to the Stack
        Args:
            members: a list containing instances of class `Member`
        """
        for member in members:
            self.add_member(member)
        self._test()

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
        assert(X.ndim <= 3)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        return self.model.fit(X, y, **kwargs)

    def predict(self, X=None, predict_proba=False, kwargs={}):
        """
        Meta-Model prediction for the class' probabilities as a regression
        problem.
        Args:
            X: input data to be predicted
            kwargs: further arguments for prediction function
            predict_proba: if should call method `predict_proba`
                instead of `predict`.
        Returns:
            the predicted probabilities as np.array
        """
        if X is None:
            X = self._get_pred_X()
        if X.ndim == 3:
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        if predict_proba and hasattr(self.model, 'predict_proba'):
            self.predictions = self.model.predict_proba(X, **kwargs)
        elif hasattr(self.model, 'predict'):
            self.predictions = self.model.predict(X, **kwargs)
        else:
            raise("Model has no predict function")
        return np.array(self.predictions)

    def describe(self, probabilities_val=None, evaluation_metric=None,
                 maximize=True):
        """
        Prints information about the performance of base and meta learners
        based on validation data.
        Args:
            probabilities_val: (optional) probabilities/prediction on
                validation data
            evaluation_metric: (optional) evaluation metric function.
                Default: `sklearn.metrics.roc_auc_score`
            maximize: if metric should be maximized (otherwise minimized)
        """
        best_score = float("-inf") if maximize else float("inf")
        if evaluation_metric is None:
            evaluation_metric = metrics.roc_auc_score
        if probabilities_val is None:
            probabilities_val = self._predict_val()
        # Assumption: all members have same val_classes
        val_classes = self.members[0].val_classes
        for i in range(self._nmembers):
            member = self.members[i]
            model_score = _calculate_metric(member.val_classes, member.val_probs, evaluation_metric)
            max_flag = maximize and model_score > best_score
            min_flag = not(maximize) and model_score < best_score
            if max_flag or min_flag:
                best_score = model_score
            text = member.name + " - {}: {:1.4f}".format(
                evaluation_metric.__name__, model_score)
            print(text)
        ensemble_score = _calculate_metric(val_classes, probabilities_val, evaluation_metric)
        print("StackEnsemble {}: {:1.4f}".format(
            evaluation_metric.__name__, ensemble_score))
        return ensemble_score

    def _get_X(self, attrname):
        X = []
        probs = getattr(self.members[0], attrname)
        # Assumption: all members have same train_probs length
        for i in range(len(probs)):
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
        Test assumption that all members' classes have same shape and values.
        Names should be unique.
        This is an internal condition for class structures.
        """
        if self._nmembers < 2:
            return True
        t1 = [(np.array_equal(self.members[i].train_classes,
                              self.members[i + 1].train_classes))
              for i in range(self._nmembers - 1)]
        t2 = [(np.array_equal(self.members[i].val_classes,
                              self.members[i + 1].val_classes))
              for i in range(self._nmembers - 1)]
        assert(np.sum(t1) == self._nmembers - 1)
        assert(np.sum(t2) == self._nmembers - 1)
        names = [self.members[i].name for i in range(self._nmembers)]
        assert(len(list(names)) == len(set(names)))
        return True

    def save(self, folder="./premodels/"):
        """
        Saves meta-learner and base-learner of ensemble into folder / directory
        Args:
            folder: the folder where models should be saved to.
                Create if not exists.
        """
        if not os.path.exists(folder):
            os.mkdir(folder)
        [member.save(folder=folder) for member in self.members]
        temp = self.members
        # Reset base-learners. These are loaded idependently
        self.members = None
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


def _calculate_metric(y_true, y_pred, metric=None):
    if metric is None:
        metric = metrics.roc_auc_score
    try:
        return metric(y_true, y_pred)
    except ValueError:
        y_true_cat = to_categorical(y_true)
        return metric(y_true_cat, y_pred)
