import numpy as np
from sklearn.base import clone
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import check_random_state
from sklearn.externals.joblib import Parallel, delayed
import numpy as np
from scipy.stats import binom_test
from sklearn.base import BaseEstimator, RegressorMixin
#from xgboost.sklearn import XGBRegressor
from functools import partial
from sklearn import preprocessing

from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Dense
from keras.models import Model
import keras.backend as K

from joblib import Parallel, delayed

class KerasDropoutPrediction(object):
    def __init__(self,model):
        self.f = K.function(
                [model.layers[0].input, 
                 K.learning_phase()],
                [model.layers[-1].output])
    def predict(self,x, n_iter=10):
        result = []
        for _ in range(n_iter):
            result.append(self.f([x , 1]))
        #result = Parallel(n_jobs=2)(delayed(self.f)([x , 1]) for i in range(n_iter))
        result = np.array(result).reshape(n_iter,len(x)).T
        return result

class NeuralNetworksDropoutRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator=None, n_jobs=1, random_state=None):
        self.random_state = random_state
        self.base_estimator = base_estimator
        self.n_jobs = n_jobs
        self.nunits = 512
        self.dropout = 0.50
        self.hidden_size = 2
        self.uq = True
        self.model = None
        self.preProcModelInput = None
        self.preProcModelOutput = None
        self.epochs = 100
        self.batch_size = 32
        
    def fit(self, X, y):
        y = np.asarray(y)
        self.preProcModelInput = preprocessing.MinMaxScaler()
        self.preProcModelInput.fit_transform(X)
        trainX = self.preProcModelInput.transform(X)

        self.preProcModelOutput = preprocessing.MinMaxScaler()
        self.preProcModelOutput.fit_transform(y.reshape(-1, 1))
        trainY = self.preProcModelOutput.transform(y.reshape(-1, 1))
        trainY = np.squeeze(np.asarray(trainY))

        input_shape = (trainX.shape[1],)
        inputs = Input(shape=input_shape)
        x = Dense(self.nunits, activation='relu')(inputs)
        x = Dropout(self.dropout)(x, training=self.uq)
        for j in range(self.hidden_size):
            x = Dense(self.nunits, activation='relu')(x)
            x = Dropout(self.dropout)(x, training=self.uq)
        level_all = Dense(1, name='output')(x)
        model = Model(inputs=inputs, outputs=level_all)
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        #model.summary() 
        self.model = model
        self.model.fit(trainX, trainY, epochs=self.epochs, batch_size=self.batch_size, verbose=0, shuffle=True)
        return self        

    def predict(self, X, return_std=False):
        testX = self.preProcModelInput.transform(X)
        mean = self.model.predict(testX)
        kdp = KerasDropoutPrediction(self.model)
        y_pred_uq = kdp.predict(testX, n_iter=10)
        y_pred_transform = np.zeros(y_pred_uq.shape)
        for r in range(y_pred_uq.shape[0]):
            k = self.preProcModelOutput.inverse_transform(y_pred_uq[r,:].reshape(-1, 1)).reshape(1, -1)[0]
            y_pred_transform[r,:] =  k[0]
        
        #mean = y_pred_uq.mean(axis=1)
        stdv = y_pred_transform.std(axis=1).reshape(-1, 1)
        #print(mean)
        #print(y_pred_transform[0,:])
        #print(stdv.shape)
        #print(mean.shape)
        if return_std:
            return mean, stdv

        # return the mean
        return mean

