import numpy as np
from sklearn import metrics
import warnings
from keras.models import Model
from keras.layers import Dense
from keras.layers.merge import concatenate
from keras import backend as K
import tensorflow as tf
from abc import abstractmethod
from sklearn.ensemble import RandomForestRegressor
import os
from keras.models import load_model


class Member:
    """
    Representation of a single member of an Ensemble
    """

    def __init__(self, model, train_batches=None, val_batches=None, pred_batches=None, name=None,
                 submission_probs=None):
        """
        Constructor of a Keras Ensemble Member for a Binary Classification (Image Recognition) Task.
        If `val_predictions` is None, then `val_predictions` will be calculated using `imggen`.
        Args:
            model: the (pre-trained) keras model
            train_batches: an instance of Keras `ImageDataGenerator` for training the Stacked Model
            val_batches: an instance of Keras `ImageDataGenerator` for validating the Stacked Model
            pred_batches: an instance of Keras `ImageDataGenerator` for prediction
            submission_probs: the submission prediction probabilities, to be weighted in case of DirichletEnsemble building
            name: name of the model
        """
        self.model=model
        self.train_batches=train_batches
        self.val_batches=val_batches
        self.pred_batches=pred_batches
        self.submission_probs=submission_probs
        self.name=name
        #Initialize Params
        self.val_probs=None
        self.train_probs=None
        self.val_classes = None
        self.train_classes = None

    def _calculate_predictions(self, batches):  # TODO: call automatically for dirichlet ensemble
        if hasattr(batches, 'shuffle'):
            batches.reset()
            batches.shuffle=False
        preds=self.model.predict_generator(batches, steps=(batches.n // 32) + 1, verbose=1)
        if preds.shape[0] > 1:
            warnings.warn("Caution! This program is still not supporting multi-class problems.")
            probs=preds[:, 0]
        else:
            probs=preds
        return probs

    def _calculate_val_predictions(self):  # TODO: call automatically for dirichlet ensemble
        self.val_probs=self._calculate_predictions(self.val_batches)
        self.val_classes=self.val_batches.classes
        return self.val_probs

    def _calculate_train_predictions(self):  # TODO: call automatically for dirichlet ensemble
        self.train_probs=self._calculate_predictions(self.train_batches)
        self.train_classes=self.train_batches.classes
        return self.train_probs

    def save(self, folder="./premodels/"):
        if not os.path.exists(folder):
            os.mkdir(folder)
        if not os.path.exists(folder+self.name):
            os.mkdir(folder+self.name)
        if self.val_probs is None:
            try:
                self.val_probs=self._calculate_val_predictions()
            except Exception as e:
                print(e)
        if self.train_probs is None:
            try:
                self.train_probs=self._calculate_train_predictions()
            except Exception as e:
                print(e)
        np.save(folder + self.name+"/val_probs.npy", self.val_probs)
        np.save(folder + self.name + "/train_probs.npy", self.train_probs)
        np.save(folder + self.name + "/val_classes.npy", self.val_batches.classes)
        np.save(folder + self.name + "/train_classes.npy", self.train_batches.classes)

    def load(self, folder="./premodel/", keras_model_path=None, keras_kwargs={}):
        self.val_probs = np.load(folder + self.name + "/val_probs.npy")
        self.train_probs = np.load(folder + self.name + "/train_probs.npy")
        self.val_classes = np.load(folder + self.name + "/val_classes.npy")
        self.train_classes = np.load(folder + self.name + "/train_classes.npy")
        self.model = load_model(keras_model_path, **keras_kwargs)


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
        if member.val_batches is not None and member.val_probs is None:
            member.val_probs = member._calculate_val_predictions()

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
        self.fitted = True

    def predict(self):
        """
        Returns the weighed probabilities of the ensemble members
        Returns:
            the predicted probabilities as np.array
        """
        if not self.fitted:
            raise("Model still not fitted. Fit the ensemble model first")
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
        result = ""
        if modelbestauc > self.bestauc:
            result = "Worse Performance. Achieved an AUC of {} but the best ensemble member " \
                     "alone achieves an AUC of {}".format(self.bestauc, modelbestauc)
        else:
            result = "Better Performance. Achieved an AUC of {} and the best ensemble member " \
                     "alone achieves an AUC of {}".format(self.bestauc, modelbestauc)
        print(result)
        return    


class StackingEnsemble:
    def __init__(self, model):
        """
        Constructor of a Keras Ensemble Binary Classifier
        Args:
            model: ensemble model which should serve as meta-model
        """
        if model is None:
            self.model = RandomForestRegressor(verbose=1, n_estimators=1000, max_depth=5)
        # Initialize Parameters:
        self.members = []
        self._nmembers = 0
        
    def _get_train_X(self):
        X = []
        for i in range(len(self.members[0].train_probs)):  # Assumption: all members have same train_probs length
            preds = []
            for member in self.members:
                preds.append(member.train_probs[i])
            X.append(preds)
        return np.array(X)

    def _get_val_X(self):
        X = []
        for i in range(len(self.members[0].val_probs)):  # Assumption: all members have same val_probs length
            preds = []
            for member in self.members:
                preds.append(member.val_probs[i])
            X.append(preds)
        return np.array(X)

    def _get_pred_X(self):     
        X = []
        for i in range(len(self.members[0].submission_probs)):
            preds = []
            for member in self.members:
                preds.append(member.submission_probs.iloc[i])
            X.append(preds)
        return np.array(X)   

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
            except:
                pass
        if member.train_probs is None:
            try:
                member.train_probs = member._calculate_train_predictions()
            except:
                pass

    def fit(self, X, y, kwargs={}):
        """
        Trains the meta-model
        Args:
            X: training data
            y: training classes
            kwargs: further arguments for the fit function
        """
        assert(len(self.members) > 1)
        # Assumption: all members have same train_batches.classes
        return self.model.fit(X, y, **kwargs)  

    def predict(self, X, kwargs={}):
        """
        Meta-Model prediction for the probabilities of the positive class as a regression problem
        Args:
            X: input data to be predicted
            kwargs: further arguments for prediction function
        Returns:
            the predicted probabilities as np.array
        """
        return self.model.predict(X, **kwargs)

    def describe(self, probabilities_val, positive_class=0):
        """
        Prints information about the ensemble members and its weights as well as single and ensemble AUC performance
        on validation dataset.
        """
        modelbestauc = 0
        val_classes = self.members[0].val_classes  # Assumption: all members have same val_classes
        for i in range(self._nmembers):
            model = self.members[i]
            auc = metrics.roc_auc_score([x == positive_class for x in val_classes], model.val_probs)
            if auc > modelbestauc:
                modelbestauc = auc
            print(self.members[i].name, auc)
        auc = metrics.roc_auc_score([x == positive_class for x in val_classes], probabilities_val)
        print("Ensemble", auc)
