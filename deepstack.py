import numpy as np
from sklearn import metrics
import warnings
from keras.models import Model
from keras.layers import Dense
from keras.layers.merge import concatenate
from keras import backend as K
import tensorflow as tf
from abc import abstractmethod


class Member:
    """
    Representation of a single member of an Ensemble
    """
    def __init__(self, model, train_batches=None, val_batches=None, pred_batches=None, name=None, submission_probs=None):
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
        self.model = model
        self.train_batches = train_batches
        self.val_batches = val_batches
        self.pred_batches = pred_batches
        if train_batches is not None and hasattr(train_batches, 'shuffle'):
            self.train_batches.shuffle = False
        if val_batches is not None and hasattr(val_batches, 'shuffle'):
            self.val_batches.shuffle = False
        if pred_batches is not None and hasattr(pred_batches, 'shuffle'):
            self.pred_batches.shuffle = False
        self.submission_probs = submission_probs
        self.name = name

    def _calculate_val_predictions(self):  # TODO: call automatically for dirichlet ensemble
        if hasattr(self.val_batches, 'shuffle'):
            self.val_batches.reset()
            self.val_batches.shuffle = False
        preds = self.model.predict_generator(self.val_batches, steps=(self.val_batches.n // 32) + 1, verbose=1)
        if preds.shape[0] > 1:
            warnings.warn("Caution! This program is still not supporting multi-class problems.")
            return preds[:,0]
        return preds


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
        if member.val_batches is not None:
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


def _combined_generators(member_batches):
    while True:
        nimgs = [next(imggen) for imggen in member_batches]
        nextImages = [nimg[0] for nimg in nimgs]
        #assert(nimgs[0][1] == nimgs[1][1])  # TODO: Test if all classes from Generators are equal
        yield nextImages, nimgs[0][1]


def auc(y_true, y_pred):
            auc = tf.metrics.auc(y_true, y_pred)[1]
            K.get_session().run(tf.local_variables_initializer())
            return auc        