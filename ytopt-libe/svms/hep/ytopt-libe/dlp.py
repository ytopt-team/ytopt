#!/usr/bin/env python

import pandas as pd
import numpy as np
import glob, os
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import resample

from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc

from sklearn import metrics
from numpy.typing import NDArray
from typing import List, TypedDict
import logging
from sklearn.pipeline import make_pipeline


# sorting the charge and label files coz indexes in diff order
def sort_files(filename):
    return int(filename.split('_d')[1].split('.csv')[0])
    

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
        )

## Includes CotBeta
#read the charge-recon files
def readCharge(file_path):
    try:
        df = pd.read_csv(file_path)
        # df = df.iloc[:,:273] #first timestamp
        return df
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}")
        return None
        
        
#read the label files
def readLabel(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}")
        return None
        
# data and labels
def data_dict(charge_files,label_files,pt_thres):
    charge_comb = pd.DataFrame()
    pt_comb = pd.DataFrame()
    y_local = pd.DataFrame()
    cotBeta = pd.DataFrame()
    
    for i in range(len(charge_files)):
        charge_curr = readCharge(charge_files[i])
        charge_comb = pd.concat([charge_comb,charge_curr],ignore_index=True)
        
        pt_curr = readLabel(label_files[i])
        pt_comb = pd.concat([pt_comb,pt_curr],ignore_index=True)
        
        y_curr = readLabel(label_files[i])['y-local']
        y_local = pd.concat([y_local,y_curr],ignore_index=True)
        
        cotbeta =readLabel(label_files[i])['cotBeta']
        cotBeta = pd.concat([cotBeta,cotbeta],ignore_index=True)
        
        
    pt_comb=pt_comb['pt']  ## select only pt columns
    
    #labeling based on pt values 
    label = []
    for value in pt_comb:
        if value > pt_thres:
            label.append(1)
        else:
            label.append(0)
            
            
            
    #check equal data and label
    if len(charge_comb)== len(label):
        dsize = len(charge_comb)
        print('labeling done')
        
        
    #convert in nparray
    charge=np.array(charge_comb)
    label=np.array(label)
    y_local=np.array(y_local)
    
    #use abs value of charge, some are small negs
    #charge_comb=charge_comb.abs()
    
    #convert in nparray
    charge_comb=np.array(charge_comb)
    label=np.array(label)
    cotBeta=np.array(cotBeta)
    
    ##using last timeslice data
    charge=charge.reshape(charge.shape[0],8,273)
    #cls_size = np.array(cluster_size(charge))
    #cls_size = cls_size[:, np.newaxis]
    
    charge = charge[:,-1:]
    charge = np.squeeze(charge,axis=1)
    
    ## applying y-profile (shruti's comment)
    charge = charge.reshape(charge.shape[0], 13,21)
    charge = charge.sum(axis=2)

    ## calculate y-cluster size [#pixels]
    cluster_size=[]
    for row in charge:
        cluster_size.append(np.count_nonzero(row>400))
    cluster_size=np.array(cluster_size)
    cluster_size=cluster_size.reshape(-1,1)
    
    #return cluster_size, label, y_local, pt_comb

    return charge,label,y_local, cotBeta

#custom kernel
class CustomKernel:
    def sigmoid_kernel(self, x: NDArray[np.float32], y: NDArray[np.float32],
                       ratio:float, coef0_param:float) -> NDArray[np.float64]:
        
        output = np.asarray(np.tanh(ratio * np.dot(x, y.T) + coef0_param))
        return output

    def euc_dist_mat(
            self, x: NDArray[np.float32],
            y: NDArray[np.float32]) -> NDArray[np.float64]:
        x_norm = (x**2).sum(axis=1)
        y_norm = (y**2).sum(axis=1)
        return np.asarray(
            np.abs(
                x_norm.reshape(-1, 1) + y_norm -
                2 * np.dot(x, y.T)))

    def gaussian_kernel(self, x: NDArray[np.float32], y: NDArray[np.float32],
                       ratio:float) -> NDArray[np.float64]:
        
        dists_sq = self.euc_dist_mat(x, y)
        output = np.asarray(np.exp(-ratio * dists_sq))

        return output

    def mixed_kernel(self,
                     x: NDArray[np.float32],
                     y: NDArray[np.float32],
                     mixing_ratio: float,
                     sigmoid_ratio: float,
                     coef0_param: float, 
                     gaussian_ratio: float) -> NDArray[np.float64]:

        return np.asarray(
                (1 - mixing_ratio) * self.sigmoid_kernel(x,y,sigmoid_ratio,coef0_param) +
                mixing_ratio * self.gaussian_kernel(x,y,gaussian_ratio))

def model_eval(X_train,y_train,X_test,y_test):
    
    #hyperparameter list
    mixing_ratio = #P0
    sigmoid_ratio = #P1
    gaussian_ratio = #P2
    coef0_param = #P3 
    C_param = #P4
    
    model= SVC(C=C_param,probability=True,kernel=lambda x, y: CustomKernel().mixed_kernel(
        x, y, mixing_ratio, sigmoid_ratio, coef0_param, gaussian_ratio),class_weight='balanced')
        
    ## fit the model and evaluate
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test) 
    
    # Get the probabilities for the positive class (label 1)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    ## confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.5f}")

    # Calculate precision
    precision = metrics.precision_score(y_test, y_pred)
    print(f"Precision: {precision:.5f}")
    
    # Calculate recall
    recall = metrics.recall_score(y_test, y_pred)
    print(f"Recall: {recall:.5f}")
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    #Calculate AUC (Area Under Curve)
    roc_auc = auc(fpr, tpr)
    #return y_pred, y_prob

def plot(y_pred,y_test,y_prob):
    
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Ensure the 'plot' directory exists
    if not os.path.exists('plot'):
        os.makedirs('plot')
	
    #Plot the Confusion Matrix
    #plt.figure(figsize=(10,7))
    #sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')  # fmt='g' avoids scientific notation
    #plt.xlabel('Predicted Label')
    #plt.ylabel('Actual Label')
    #plt.title('Confusion Matrix')
    #filename = "confusion_matrix_plot.png"
    #filepath = os.path.join('plot', filename)
    #plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    #plt.show()
    
    
    # Calculate accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.5f}")
    
    # Calculate precision
    precision = metrics.precision_score(y_test, y_pred)
    print(f"Precision: {precision:.5f}")
    
    # Calculate recall
    recall = metrics.recall_score(y_test, y_pred)
    print(f"Recall: {recall:.5f}")
    
    # Calculate F1 score
    f1 = metrics.f1_score(y_test, y_pred)
    print(f"F1Score: {f1:.5f}")
    print('fp: ', cm[0][1]/len(y_test))
    print('fn: ',cm[1][0]/len(y_test))
    print('rejection: ',(cm[0][0]+cm[1][0])/len(y_test))
    
    
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    # Calculate AUC (Area Under Curve)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    #plt.figure(figsize=(10, 7))
    #plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.5f})')
    #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # diagonal line
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('Receiver Operating Characteristic (ROC) Curve')
    #plt.legend(loc='lower right')
    #plt.grid(True)
    #filename = "roc_plot.png"
    #filepath = os.path.join('plot', filename)
    #plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    #plt.show()
    
  
def main():
    ## data
    path = '/Users/xingfu/research/tmp/ytune/ytopt-libensemble/ytopt-libe-svms/hep/positive-charge/'
    charge_files = glob.glob(path+'recon8t_d1695*')
    label_files= glob.glob(path+'labels_d1695*')
    
    # data files		  
    charge_files = sorted(charge_files, key=sort_files)
    label_files = sorted(label_files,key=sort_files)
    
    #select only few files
    label_files=label_files[:1]
    charge_files=charge_files[:1]
    
    #define threshold
    pt_thres = 0.2
    
    ## unpack the data
    charge,label,y_local,cotBeta = data_dict(charge_files,label_files,pt_thres) 
    
    
    #just for checking, applying threshold of 400e
    charge[charge<400]=0
    
    # y_profile+y_local information
    charge=np.hstack((charge,y_local)) 
    
    # charge=np.hstack((charge,y_local))
    print(f'charge shape: {charge.shape}')
    
    #split data
    X_train, X_test, y_train, y_test = train_test_split(charge,label,test_size=0.3,shuffle=True,random_state=42)
    
    #standarization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

   
    #retrive the information
    #y_pred, y_prob = model_eval(X_train,y_train,X_test) 
    
    model_eval(X_train,y_train,X_test,y_test)

    #plot
    #plot(y_pred,y_test,y_prob)
   

if __name__ == "__main__":
    main()
    
