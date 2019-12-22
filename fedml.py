from abc import ABC, abstractmethod
import pickle
import os
from utils import compute_errorRate
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from classification_dataset_preprocessing import *
import partition_dataset as partitions
import uuid
from collections import OrderedDict
from collections import Counter
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from random import sample
import copy
import math
import copy
import psutil
import time


import keras
# from fyrai.runtime.runtime import Runtime
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

# Create an initial global CNN Model
class BaseLearner(ABC):

    def  __init__(self):
        """ fjlksdjfgksj """

    @abstractmethod    
    def average_weights(self, model2):
        print("Must be implemented by subclass")

    @abstractmethod
    def partial_fit(self,x,y):
        print("Must be implemented by subclass")

    @abstractmethod
    def predict(self,x):
        print("Must be implemented by subclass")


class KerasSequentialBaseLearner(BaseLearner):
    """  Keras Sequential base learner."""

    def __init__(self, model=None):
        self.model = model

    @staticmethod
    def average_weights(models):
        """ fdfdsfs """
        weights = [model.model.get_weights() for model in models]

        avg_w = []
        for l in range(len(weights[0])):
            lay_l = np.array([w[l] for w in weights])
            weight_l_avg = np.mean(lay_l,0)
            avg_w.append(weight_l_avg)

        return avg_w

    def set_weights(self,weights):
        self.model.set_weights(weights)

    def partial_fit(self,x,y,classes=None):
        """ Do a partial fit. """
        batch_size = 32
        epochs = 1
        self.model.fit(x,y,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                )

    def predict(self,x):
        y = self.model.predict(x)




class SGDBaseLearner(BaseLearner):
    """ sklearn SGDClassifier base learner. """

    def  __init__(self,classes=None):
        penalty = 'l2'
        alpha = 1e-3
        self.model = SGDClassifier(loss='hinge',penalty=penalty,l1_ratio = 0,
                                    alpha=alpha, max_iter=100,  warm_start=False)
        self.model.classes_ = classes

    @staticmethod
    def average_weights(models):

        for model in models:
            try:
                coef += model.model.coef_
                intercept += model.model.intercept_
            except:
                coef = model.model.coef_
                intercept = model.model.intercept_
        coef = coef/len(models)
        intercept = intercept/len(models)
        weights={}
        weights['coef'] = coef
        weights['intercept'] = intercept
        return weights

    def set_weights(self,weights):
        self.model.coef_ = weights['coef']
        self.model.intercept_ = weights['intercept']


    def partial_fit(self,x,y,classes=None):
        self.model.partial_fit(x,y,classes=classes)

    def predict(self,x_test):
        return self.model.predict(x_test)


class AllianceModel(ABC):
    """ An AllianceModel is an implementation of a training scheme running on top of an Alliance. """

    def __init__(self, alliance):
        self.alliance = alliance

    @abstractmethod
    def fit(self, parameters=None):
        """This is the 'orchestrator' """
        print("Must be implemented by subclass")

    @abstractmethod
    def predict(self, parameters=None):
        print("Must be implemented by subclass")


class PartialIncrementalLearnerClassifier(AllianceModel):
    """ Train a model using partial incremental  learning. """

    def __init__(self, alliance, base_learner = None):
        self.base_learner = base_learner
        self.current_global_model = None
        super().__init__(alliance)

    def fit(self, parameters = None):
        """  """
        training_loss = []
        test_loss = []

        if self.base_learner is None:
            self.base_learner = SGDBaseLearner()

        if not self.current_global_model:
            self.current_global_model  = self.base_learner

        #  Start training 
        for j in range(parameters["nr_global_iterations"]):

            partialModel = copy.deepcopy(self.current_global_model)

            rand_indx = np.random.permutation(len(self.alliance.members))
            for indx in rand_indx:
                self.alliance.members[indx].train(partialModel,nr_iter=parameters["nr_local_iterations"])

            # Training loss, mean error rate over all alliance training data
            tl = self.alliance.alliance_training_loss(partialModel)
            training_loss.append(tl)
            # Test loss, mean error rate on a  validation set
            try:
                test_loss.append(self.alliance.alliance_test_loss(partialModel))
                print("test_loss : ", test_loss)
                # TODO: Implement early stopping
            except:
                pass
                
            # Update the global model
            # TODO: investigate accept-reject schemes (model updates vs. risk of getting stuck in local minima)
            self.current_global_model = partialModel

        return training_loss, test_loss  

    def predict(self,x_test):
        """ fdfsd """ 
        return self.current_global_model.predict(x_test)


class BaggingPartialIncrementalLearnerClassifier(AllianceModel):
    """ Bagging PIL classifier. """
    def __init__(self, alliance, base_learner = None):
        self.base_learner = base_learner
        self.current_global_model = None
        super().__init__(alliance)

    def fit(self, parameters = None):
        """ Train an ensemble of number_of_models submodels, each a partial incremental learner
        based  on a subset of n_members alliance members. """

        if self.base_learner is None:
            self.base_learner = SGDBaseLearner()

        if not self.current_global_model:
            self.current_global_model = self.base_learner


        n_members = len(self.alliance.members)

        model = {'parameters':parameters,'n_members':n_members}
        model['models'] = {}

        training_loss = []
        test_loss = []

        # TODO: do in parallel 
        for i in range(parameters["number_of_models"]):

            partialModel = copy.deepcopy(self.base_learner)

            # shuffle the index first
            randIndex = sample(list(range(n_members)), n_members)
            # Sample the members participating in this learner
            trainIndex = sample(randIndex, int(parameters["member_fraction"]*n_members))

            for j in range(parameters["nr_global_iterations"]):
                for indx in trainIndex:
                    self.alliance.members[indx].train(partialModel, nr_iter=parameters["nr_local_iterations"])

            # Out-of-bag training loss for submodel
            lacc = 0.0
            validation_members=list(set(randIndex) - set(trainIndex))
            for mindx in validation_members:
                lacc += 1.0 - self.alliance.members[mindx].scoreLocalData(partialModel)
            w = lacc/len(validation_members)

            model['models'][i] = {'partialModel':partialModel, "members":trainIndex, "oob_loss":w}
            self.current_global_model = model
            # Training loss, mean error rate over all alliance training data
            tl = self.alliance.alliance_training_loss(self)
            training_loss.append(tl)
            # Test loss, mean error rate on a  validation set
            try:
                test_loss.append(self.alliance.alliance_test_loss(self))
                # TODO: Implement early stopping
            except:
                pass

        self.current_global_model = model

        return training_loss, test_loss


    def predict(self, x_test,type='hard'):
        """ Voting classifier (hard voting). """
        if type == 'hard':
            votes = []
            for model_id, model in self.current_global_model['models'].items():
                z = model['partialModel'].predict(x_test)
                w = model['oob_loss']
                votes.append(z)

            votes = np.array(votes)
            (nvoters, npoints) = np.shape(votes)

            y_pred = []
            for i in range(npoints):
                bins = np.bincount(votes[:, i])
                y_pred.append(np.argmax(bins))

            y_pred = np.array(y_pred)
        return y_pred


class FedAveragingClassifier(AllianceModel):
    """  Difference here is that  we need to average parameters/weights in each iteration.
    Becomes ML framework (scikit-learn, Keras etc) dependent.  """ 
    def __init__(self, alliance, base_learner = None, name='example'):

        if base_learner is None:
            penalty = 'l2'
            alpha = 1e-3
            base_learner = SGDClassifier(loss='hinge',penalty=penalty,l1_ratio = 0,
                                            alpha=alpha, max_iter=100,  warm_start=False)
        self.base_learner = base_learner
        self.current_global_model = None
        self.default_parameters = {"nr_global_iterations":100, "nr_local_iterations":1, "training_steps":None}
        self.weights_std = []
        self.training_loss = []
        self.test_loss = []
        nr = ""
        while os.path.exists('test_loss_' + name + str(nr) + '.p'):
            if nr == "":
                nr = 1
            else:
                nr += 1
        self.filename = 'test_loss_' + name + str(nr) + '.p'



        super().__init__(alliance)


    def fit(self, parameters=None):

        """  """

        # fill default values in parameters
        if not "nr_global_iterations" in parameters:
            parameters["nr_global_iterations"] = 10
        if not "training_steps" in parameters:
            parameters["training_steps"] = None
        if not "c_parameter" in parameters:
            parameters["c_parameter"] = len(self.alliance.members)

        if not self.current_global_model:
            self.current_global_model = self.base_learner

        if not self.alliance.temp_model:
            self.alliance.temp_model = self.base_learner

        for member in self.alliance.members:
            member.set_model(copy.deepcopy(self.current_global_model))
        #  Start training 
        for j in range(parameters["nr_global_iterations"]):
            print("global epoch: ", j)
            print("virtual memory used: ", psutil.virtual_memory()[2], "%")

            # This step is a map operation - should happen in parallel/async
            rand_indx = np.random.permutation(len(self.alliance.members))[:parameters["c_parameter"]]
            global_weights = self.current_global_model.model.get_weights()

            for indx in rand_indx:

                self.alliance.members[indx].model.set_weights(global_weights)
                self.alliance.members[indx].train(self.alliance.members[indx].model,
                                                  parameters=parameters)

            # Average the model updates  - here  we have a global synchronization step. Server should aggregate
            if parameters['model_size_averaging'] == True:
                temp_data = np.array([[member.model, member.data_size] for member in self.alliance.members])
                all_models = list(temp_data[:,0])
                parameters['model_sizes'] = list(temp_data[:,1])
                new_weights, weights_std = self.current_global_model.average_weights(all_models, parameters)
            else:
                all_models = [member.model for member in self.alliance.members]
                new_weights, weights_std = self.current_global_model.average_weights(all_models,parameters)

            self.current_global_model.set_weights(new_weights)
            self.training_loss.append(self.alliance.alliance_training_loss(self.current_global_model))

            # Test loss, mean error rate on a  validation set
            try:
                self.test_loss.append(self.alliance.alliance_test_loss(self.current_global_model))
                pickle.dump(self.test_loss, open(self.filename, 'wb'))

                # TODO: Implement early stopping
            except:
                pass


            print("test_loss: ", np.round(np.array(self.test_loss), 3))

        return self.test_loss

    def predict(self,x_test):
        """ fdfsd """ 
        return self.current_global_model.predict(x_test)

    def global_score_local_models(self):

        # average all models score
        model_members = [self.alliance.members[m].model for m in
                         list(set(np.arange(len(self.members))))]

        w, _ = self.temp_model.average_weights(model_members)
        self.current_global_model.set_weights(w)
        test_loss_all = self.alliance_test_loss(self.temp_model)
        best_w = w
        print("test loss all: ", np.round(test_loss_all, 4))
        best_loss = test_loss_all
        for model_member in range(len(self.alliance.members)):
            print("model ", self.alliance.members[model_member].data_size, " starts:")
            # self.alliance.members[model_member].score_test_set.append(self.alliance_test_loss(self.members[model_member].model))
            model_members = [self.members[m].model for m in
                             list(set(np.arange(len(self.members))) - set([model_member]))]

            w, _ = self.temp_model.average_weights(model_members)
            self.current_global_model.set_weights(w)
            test_loss_wo = self.alliance_test_loss(self.current_global_model)
            print("test loss wo: ", np.round(test_loss_wo, 4))

            q_score = self.test_loss_all[-1] - test_loss_wo
            self.members[model_member].q_score.append(q_score)
            if test_loss_wo > best_loss:
                best_loss = test_loss_wo
                best_w = w

        self.current_global_model.set_weights(best_w)


class Alliance(object):
    """ The server who coordinates """

    def __init__(self, penalty='l2', classes=None, members=None):
        """ """
        self.members = []
        self.currGlobalModel = None #current global model
        self.temp_model = None
        self.penalty = penalty
        self.classes = classes
        self.delta_glob_weights = []
        self.test_loss = []
        self.test_loss_all = []

    def add_member(self, member): # and register
        self.members.append(member)
        #print("Register alliance member: {0}".format(member.id))

    def set_classes(self, classes): 
        """ Set list of all possible classes globally (needed for multi-label classification) """
        self.classes = classes

    def set_validation_dataset(self, x_test,y_test):
        self.x_test = x_test
        self.y_test = y_test 

             
    def predictGlobalModel(self,x_test,model=None):
        return model.predict(x_test)


    def __globalSGDModelCV(self):
        penalty = ['l2']
        alpha = [0.001, 0.0001, 0.00001, 0.000001, 1e-7, 1e-8]
        score = np.zeros((len(penalty),len(alpha)))

        for p in range(len(penalty)):
            for a in range(len(alpha)):
                partialModel = SGDClassifier(loss='hinge', l1_ratio = 0, warm_start = False,
                                            penalty=penalty[p], alpha=alpha[a], max_iter=100)

                #for i in range(100):
                for m in range(len(self.members)//2):
                    self.members[m].trainGlobalModel(partialModel)
                s = 0
                for m in range(len(self.members)//2,len(self.members)):
                    s += self.members[m].scoreLocalData(partialModel)
                # two fold CV

                partialModel = SGDClassifier(loss='hinge', l1_ratio=0, warm_start=False,
                                             penalty=penalty[p], alpha=alpha[a], max_iter=100)

                #for i in range(100):
                for m in range(len(self.members) // 2, len(self.members)):
                    self.members[m].trainGlobalModel(partialModel)

                for m in range(len(self.members) // 2):
                    s += self.members[m].scoreLocalData(partialModel)

                score[p,a] = s

        #Best parameters set
        index =  np.argmin(score)
        pen = penalty[index//len(alpha)]
        al = alpha[index%len(alpha)]
        return pen, al


    def alliance_training_loss(self,alliance_model):
        member_loss = []
        for member in self.members: 
            member_loss.append(member.scoreLocalData(alliance_model))
        return np.mean(member_loss)

    def alliance_test_loss(self,alliance_model):
        """ Use alliance global validation data.  """
        print("alliance_test_loss")
        y_pred = alliance_model.predict(self.x_test)
        error_rate = compute_errorRate(self.y_test, y_pred)
        return  1 - error_rate/2

    def errRateGlobalModel(self, x_test, y_test,model=None):
        if not model:
            model = self.trainGlobalModel()
        y_pred = self.predictGlobalModel(x_test,model)
        errRate = compute_errorRate(y_test, y_pred)
        return errRate

    def errRateFedEnsembleGlobalModel(self, x_test, y_test,model=None):
        if not model:
           model = self.trainEnsemble(x_test)
        y_pred = self.predictFedEnsembleGlobalModel(model,x_test)
        errRate = compute_errorRate(y_test, y_pred)
        return errRate

    def errRateEnsembleModel(self, x_test, y_test):
        y_pred = self.predictEnsemble(x_test)
        if y_pred is None:
            return np.nan
        errRate = compute_errorRate(y_test, y_pred)
        return errRate

    def global_score_local_models(self):

        print("test loss all[-1]: ", np.round(self.test_loss_all[-1],4))

        for model_member in range(len(self.members)):
            print("model ", self.members[model_member].data_size, " starts:")
            self.members[model_member].score_test_set.append(self.alliance_test_loss(self.members[model_member].model))
            model_members = [self.members[m].model for m in list(set(np.arange(len(self.members))) - set([model_member]))]

            # if self.temp_model is None:
            #     self.temp_model = copy.deepcopy(self.currGlobalModel)

            w,_ = self.temp_model.average_weights(model_members)
            self.temp_model.set_weights(w)
            test_loss_wo = self.alliance_test_loss(self.temp_model)
            print("test loss wo: ", np.round(test_loss_wo,4))
            q_score = self.test_loss_all[-1] - test_loss_wo
            self.members[model_member].q_score.append(q_score)




class AllianceMember(object):
    """ Member of machine learning alliance """

    def __init__(self, x_train, y_train, classes=None):
        """ """
        self.id = uuid.uuid4()
        # Private data
        self.__x_train = x_train
        self.__y_train = y_train

        self.model = None
        self.P = x_train.shape[1]
        self.loss = 'hinge'
        self.classes = classes
        self.global_score = []
        self.data_set_index = 0
        self.data_order = np.arange(len(x_train))
        self.data_size = len(x_train)
        self.score_test_set = []
        self.delta_weights = []
        self.weights_spread = []
        self.q_score = []

    def get_model(self):
        if self.model is None:
            self.model = self._train_local_models()
        return self.model

    def set_model(self, model):
        self.model = model

    def set_classes(self, classes): 
        self.classes = classes

    ### Train ###

    def _train_local_models(self, loss='hinge'):
        model = None
        return self.__localSGDModel()

    def __localSGDModel(self):
        """ Train an SGD model on the local data of this alliance member. 
            Simple  gridsearch for  hyperparameter tuning. """

        parameters = {
            #'loss': ('log', 'hinge'),
            'penalty': ['l2'],
            'alpha': [0.001, 0.0001, 0.00001, 0.000001]
        }

        try:
            model = SGDClassifier(loss='hinge', max_iter=100, l1_ratio = 0)
            grid_search = GridSearchCV(model, parameters,cv=5,iid=False)
            grid_search.fit(self.__x_train, self.__y_train)
            # Best parameters
            best_parameters = grid_search.best_estimator_.get_params()

            penalty = best_parameters['penalty']
            alpha = best_parameters['alpha']
            model = SGDClassifier(loss='hinge', penalty=penalty, alpha=alpha, l1_ratio=0, max_iter=100)
            model.fit(self.__x_train, self.__y_train)
            self.model = model

        except:
            # insufficient data to learn a local model
            self.model = None

        return self.model

    def train(self, partialModel, nr_iter=1, parameters=None): # training_steps=None, data_augmentation=True,
             # batch_size=32, learning_rate=0.001, decay=0):
        """ Update global model by training nr_iter iterations on local training data. """

        for j in range(nr_iter):


            data_set_index, data_order = partialModel.partial_fit(x=self.__x_train,
                                                                  y=self.__y_train,
                                                                  classes=self.classes,
                                                                  data_set_index=self.data_set_index,
                                                                  data_order=self.data_order,
                                                                  parameters=parameters)

            self.data_set_index = data_set_index
            self.data_order = data_order


    ### Predict ###


    def predict(self, x_test):
        """ """
        if self.model is None:
            self.model = self._train_local_models()

        # if no model was trained due to insufficient data, return none
        if self.model is None:
            return None

        return self.model.predict(x_test)


    def predict_prob(self, x_test):
        """ """
        if self.model is None:
            self.model = self._train_local_models()

        return self.model.predict_proba(x_test)

    ###  Validate  ###

    def errRate(self, x_test, y_test):
        """ Error rate for  this members local data.  """
        y_pred = self.predict(x_test)
        if y_pred is None:
            return np.nan
        err_rate = compute_errorRate(y_test, y_pred)
        return err_rate

    def scoreLocalData(self, partial_model):
        """ Error rate for partial_model/global model on this members local training data.  """
        if partial_model is None:
            return 1;
        y_pred = partial_model.predict(self.__x_train)
        # validation = partial_model.model.evaluate(self.__x_train,self.__y_train)
        errRate = compute_errorRate(self.__y_train, y_pred)
        # print("errRate: ", errRate, "validation: ", validation)
        return 1 - errRate/2


def _split_and_scale(x,y,test_size=0.2):
    """  Split a datset x,y into training  and test data, and scale with StandardScaler """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    classes = np.unique(y)
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return classes,x_train,y_train,x_test,y_test

def _init_alliance(M, x_train, y_train,classes):

    # Split the large training dataset (x,y) in M disjoint chunks to simulate local trainers
    list_part = partitions.equal_partition(len(y_train), M)
    #list_part = partitions.unbalanced_partition(len(y_train), M)


    alliance = Alliance()
    alliance.set_classes(classes=classes)

    # Initialize alliance 
    for part_index in list_part:
        member = AllianceMember(x_train[part_index], y_train[part_index], classes=classes)
        alliance.add_member(member)

    return alliance

def run_PIL(x,y,M,parameters=None,n_repeats = 1):
    """ Experiment with hyperparameters for the federated SGD model """
    classes,x_train,y_train,x_test,y_test = _split_and_scale(x,y)
    alliance = _init_alliance(M,x_train,y_train,classes)

    # Learn a federated alliance model
    if parameters == None:
        parameters = {"nr_global_iterations": 100, "nr_local_iterations":1} 

    scores = []
    for i in range(n_repeats):
        classes,x_train,y_train,x_test,y_test = _split_and_scale(x,y)
        alliance = _init_alliance(M,x_train,y_train,classes)
        pil_model = PartialIncrementalLearnerClassifier(alliance=alliance)
        alliance.set_validation_dataset(x_test,y_test)
        training_loss,test_loss = pil_model.fit(parameters=parameters)
        scores.append(alliance.errRateGlobalModel(x_test, y_test, model = pil_model))

    return scores, training_loss, test_loss 

def run_baggingPIL(x,y,M,parameters=None,n_repeats = 1):
    """ Experiment with hyperparameters for the federated SGD model """
    classes,x_train,y_train,x_test,y_test = _split_and_scale(x,y)
    alliance = _init_alliance(M,x_train,y_train,classes)

    # Learn a federated alliance model
    if parameters == None:
        parameters = {"nr_global_iterations": 100, "nr_local_iterations":1} 

    scores = []
    for i in range(n_repeats):
        classes,x_train,y_train,x_test,y_test = _split_and_scale(x,y)
        alliance = _init_alliance(M,x_train,y_train,classes)
        bagging_pil_model = BaggingPartialIncrementalLearnerClassifier(alliance=alliance)
        alliance.set_validation_dataset(x_test,y_test)
        training_loss,test_loss = bagging_pil_model.fit(parameters=parameters)
        #scores.append(alliance.errRateGlobalModel(x_test, y_test, model = bagging_pil_model))

    return scores, training_loss, test_loss 

def run_FedAveraging(x,y,M,parameters=None,n_repeats = 1):
    """ Experiment with hyperparameters for the federated SGD model """
    classes,x_train,y_train,x_test,y_test = _split_and_scale(x,y)
    alliance = _init_alliance(M,x_train,y_train,classes)

    # Learn a federated alliance model
    if parameters == None:
        parameters = {"nr_global_iterations": 100, "nr_local_iterations":1} 

    scores = []
    for i in range(n_repeats):
        classes,x_train,y_train,x_test,y_test = _split_and_scale(x,y)
        alliance = _init_alliance(M,x_train,y_train,classes)
        base_learner = SGDBaseLearner(classes=classes)
        fed_averaging_model = FedAveragingClassifier(alliance=alliance,base_learner=base_learner)
        alliance.set_validation_dataset(x_test,y_test)
        training_loss,test_loss = fed_averaging_model.fit(parameters=parameters)
        #scores.append(alliance.errRateGlobalModel(x_test, y_test, model = bagging_pil_model))

    return scores, training_loss, test_loss 

def weights_dist(weights1, weights2):
    delta_w = []
    for w1, w2 in zip(weights1, weights2):
        delta_w.append(np.mean(abs(w1 - w2)))
        # print("delta_w shape: ", abs(w1 - w2).shape)

    return np.mean(np.array(delta_w))

def tune_federated_averaging(x,y,M):
    """ Experiment with hyperparameters for the federated SGD model """
    classes,x_train,y_train,x_test,y_test = _split_and_scale(x,y)
    alliance = _init_alliance(M,x_train,y_train,classes)
    

    # Learn a federated alliance model
    member_fraction = [0.1, 0.2, 0.3, 0.5]
    nr_models = [0.2*M, 0.5*M, M, 2*M, 10*M]

    N = 10
    for mf in member_fraction:
        parameters = {"member_fraction":mf, "number_of_models":M,"nr_global_iterations": 100, "nr_local_iterations":1} 
        scores = []
        for i in range(N):
            classes,x_train,y_train,x_test,y_test = _split_and_scale(x,y)
            alliance = _init_alliance(M,x_train,y_train,classes)
            model = alliance.trainEnsemble(parameters =  parameters)
            scores.append(alliance.errRateFedEnsembleGlobalModel(x_test, y_test, model = model))

        print(mf,np.mean(scores),1.96*np.std(scores)/math.sqrt(N))

    return scores


def run_experiment(x, y, M=5):
    global size_local_data
    # We create a global holdout set (x_validate, y_validate) for final verification
    # (this validation set might not exist in a real-life setting)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    classes = np.unique(y)
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    if M > len(x_train):
        results = OrderedDict()
        results["best_local_model_score_holdout"] = np.nan
        results["worst_local_model_score_holdout"] = np.nan
        results["mean_local_model_score_holdout"] = np.nan
        results["federated_averaging_score_holdout"] = np.nan
        results["bagging_federated_averaging_score_holdout"] = np.nan
        size_local_data.append([])
        return results

    # Split the large training dataset (x,y) in M disjoint chunks to simulate local trainers
    list_part = partitions.equal_partition(len(y_train), M)

    alliance = Alliance()
    alliance.set_classes(classes=classes)

    # Initialize alliance 
    for part_index in list_part:
        member = AllianceMember(x_train[part_index], y_train[part_index], classes=classes)
        alliance.add_member(member)

    # Score of local models on public validation set
    local_scores = []
    for member in alliance.members:
        local_scores.append(member.errRate(x_test, y_test))

    # Learn a federated alliance model
    parameters = {"member_fraction":0.25, "number_of_models":M,"nr_global_iterations": 1000, "nr_local_iterations":1} 
    model = alliance.trainEnsemble(parameters =  parameters)

    estimate_contributions = False
    if estimate_contributions: 
    	scores = []
    	all_score = 1.0-alliance.errRateFedEnsembleGlobalModel(x_test, y_test, model = model)
    	scores.append(all_score)
    	print(all_score)
    	n_models = model['M']
    	n_members = len(alliance.members)

    	member_matrix = np.zeros(shape=(n_models,n_members))
    	for j in range(n_members):
        	member_matrix[0,j]=1.0
    	for model_id, m in model['models'].items():
    		for j in m['members']:
    			member_matrix[model_id,j] = 1 

    	model_matrix  = np.ones(shape=(n_models+1,n_models))
    	for i in range(n_models):
    		m1 = copy.deepcopy(model)
    		m1['models'].pop(i)
    		#print(m1)
    		scores.append(1.0-alliance.errRateFedEnsembleGlobalModel(x_test,y_test,model=m1))
    		model_matrix[i,i]=0.0

    	scores=np.array(scores)

    	for i,s in enumerate(scores-all_score):
    		if s  > 0.0:
	    		print(model['models'][i]['members'])

    	print(scores)
    	print(all_score-scores)
    	x,residual,rank,s = np.linalg.lstsq(model_matrix,scores)
    	print(x)
    	print(np.sum(x))
    	w,residual,rank,s = np.linalg.lstsq(member_matrix,x,rcond=None)
    	print(w)
    	print(np.sum(w))

    results = OrderedDict()
    results["best_local_model_score_holdout"] = np.nanmin(local_scores)
    results["worst_local_model_score_holdout"] = np.nanmax(local_scores)
    results["mean_local_model_score_holdout"] = np.nanmean(local_scores)
    results["federated_averaging_score_holdout"] = alliance.errRateGlobalModel(x_test, y_test)
    results["bagging_federated_averaging_score_holdout"] = alliance.errRateFedEnsembleGlobalModel(x_test, y_test, model = model)

    #if boxplot_local:
        #if len(size_local_data) == 0:
    #     size_local_data.append(local_scores) #append only once for plotting

    return results


if __name__ == '__main__':
    boxplot_local = False

    #alliance_size = [2, 4, 8, 16, 32, 64, 128]
    #alliance_size = [6, 12, 18, 24]

    # for digits
    alliance_size = [20]
    #alliance_size = [20, 30, 40]

    size_local_data.clear()
    
    #dataset_name = "covertype"
    #x, y = load_covertype_dataset()

    dataset_name = "sb"
    x, y = load_spambase_data()


    #dataset_name = "bc"
    #x, y = load_breast_cancer_data()

    #dataset_name = "digits"
    #x, y = load_digit_data()

    results = OrderedDict()
    N = 1

    for M in alliance_size:
        result = Counter()
        for i in range(N):
            result = result + Counter(run_experiment(x, y, M))
        results['{0}'.format(M)] = result
        for key, value in result.items():
            result[key] = value / N

    import json
    file_name = "json_fed_ensemble/"+dataset_name+"_test_N_20.json"
    with open(file_name, 'w') as fh:
        fh.write(json.dumps(results))

    print(result)

    if boxplot_local:
        if len(size_local_data):

            #plt.ylim([0, 0.8])
            plt.xlim([0, len(alliance_size)+2])

            #plt.boxplot(size_local_data, positions=range(1,len(alliance_size)+1))
            for pos in range(len(size_local_data)):
                local_list = [x for x in size_local_data[pos] if ~np.isnan(x)]
                if len(local_list):
                    plt.boxplot(local_list, positions = [pos+1])
            # plt.boxplot(mean_local)
            xticks = ['0']
            [xticks.append(str(x)) for x in alliance_size]
            #xticks.append("")
            plt.xticks(range(len(alliance_size)+2), xticks)

            #plt.show()
            plt.savefig("localPlots/"+dataset_name+".png")
            plt.clf()


