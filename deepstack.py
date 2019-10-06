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
import joblib
import glob


class Member:
    """
    Representation of a single keras model member of an Ensemble
    """
    def __init__(self, model=None, name=None, train_batches=None, val_batches=None, pred_batches=None,
                 submission_probs=None):
        """
        Constructor of a Keras Ensemble Member for a Binary Classification (Image Recognition) Task.
        If `val_predictions` is None, then `val_predictions` will be calculated using `imggen`.
        Args:
            model: the (pre-trained) keras model
            name: name of the model. Must be unique.
            train_batches: an instance of Keras `ImageDataGenerator` for training the Stacked Model
            val_batches: an instance of Keras `ImageDataGenerator` for validating the Stacked Model
            pred_batches: an instance of Keras `ImageDataGenerator` for prediction
            submission_probs: the submission prediction probabilities, to be weighted in case of DirichletEnsemble building
        """
        assert(name is not None)
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
        self._keras_modelpath = None
        self._keras_kwargs = None

    def _calculate_predictions(self, batches):  # TODO: call automatically for dirichlet ensemble
        if hasattr(batches, 'shuffle'):
            batches.reset()
            batches.shuffle=False
        preds=self.model.predict_generator(batches, steps=(batches.n // 32) + 1, verbose=1)
        if preds.shape[0] > 1:
            print("Caution! This program is still not supporting multi-class problems.")
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

    def load_kerasmodel(self, keras_modelpath=None, keras_kwargs={}):
        self.model = load_model(keras_modelpath, **keras_kwargs)
        self._keras_modelpath = keras_modelpath
        self._keras_kwargs = keras_kwargs
        print("Keras Model Loaded:", keras_modelpath)
    
    def _load_keras(self):
        return self.load_kerasmodel(self._keras_modelpath, self._keras_kwargs)

    def save(self, folder="./premodels/", save_kerasmodel = False):  # TODO: Document
        """
        Saves member to folder
        Args:
            folder: the folder where models should be saved to. Create if not exists.
            save_kerasmodel: if it should save the keras model as part of object. 
                Recommendation is to load keras separately with method `load_kerasmodel`
        """
        if folder[-1] !=  os.sep:
            folder +=  os.sep
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
        if  save_kerasmodel:
            joblib.dump(self, os.path.join(folder+self.name, "member.joblib"))
        else:
            temp = self.model
            self.model = None  # Remove Keras model from variable
            joblib.dump(self, os.path.join(folder+self.name, "member.joblib"))
            self.model = temp

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
        if member.model is None:
            try:
                member._load_keras()
            except Exception as e:
                warnings.warn(str(e))
        return member


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
            self.predictions = self.model.predict_proba(X, **kwargs)[:,0]
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
                valprobs = [1-x for x in valprobs]
            auc = metrics.roc_auc_score(member.val_classes, valprobs)
            if auc > modelbestauc:
                modelbestauc = auc
            print(member.name,"AUC:", auc)
        auc = metrics.roc_auc_score(val_classes, probabilities_val)
        if auc < 0.5:
            probabilities_val = [1-x for x in probabilities_val]
        auc = metrics.roc_auc_score(val_classes, probabilities_val)
        print("Ensemble AUC:", auc)        

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
        X = np.concatenate((X1,X2))
        y = np.concatenate((y1,y2))
        return self.fit(X, y)

    def _predict_val(self):
        return self.predict(self._get_val_X())
    
    def _test_arrays(self, a1, a2):
        assert(a1.shape == a2.shape)
        return np.sum(a1==a2) == len(a1)  # All elements of array are equal

    def _test(self):
        """
        Test assumption that all members' classes have same shape and values. Names should be unique.
        This is an internal condition for class structures.
        """
        t1=[(self._test_arrays(self.members[i].train_classes, self.members[i+1].train_classes)) for i in range(self._nmembers-1)]
        t2=[(self._test_arrays(self.members[i].val_classes, self.members[i+1].val_classes)) for i in range(self._nmembers-1)]
        assert(np.sum(t1) == self._nmembers-1)
        assert(np.sum(t2) == self._nmembers-1)
        names = [self.members[i].name for i in range(self._nmembers)]
        assert(len(list(names)) == len(set(names)))
        return True
    
    def save(self, folder="./premodels/", save_kerasmodel = False):  # TODO: Document
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
        self.members = []  # Reset base-learners. These are loaded idependently
        self._nmembers = 0
        joblib.dump(self, os.path.join(folder, "stackensemble.joblib"))
        self.members = temp
    
    @classmethod
    def load(cls, folder="./premodels/"): 
        """
        Lods models from folder / directory
        Args:
            folder: directory where models should be loaded from
        Returns:
            loaded StackEnsemble
        """
        stack = joblib.load(os.path.join(folder, "stackensemble.joblib"))
        if folder[-1] !=  os.sep:
            folder +=  os.sep
        for fn in glob.glob(folder+"**/"):
            member = Member.load(fn)
            stack.add_member(member)
        return stack