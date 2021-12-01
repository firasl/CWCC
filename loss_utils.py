#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 15:41:30 2020

@author: laakom
"""

import tensorflow as tf
from tensorflow.keras import backend as K

def angluar_error_nll2(y_trues, y_preds):
   y_pred = y_preds[:, 0:3]
   
   #log rgb to rgb:
   #y_pred = log_to_rgb(-y_pred)
   #y_true  = log_to_rgb(-y_trues)   
   #arccoss error
   deff =  tf.acos( K.clip(K.sum( tf.nn.l2_normalize(y_trues ,axis = -1)*tf.nn.l2_normalize(y_pred ,axis = -1) ,axis=-1,keepdims=True ) , 0.0001,  1-0.0001))

   return  K.mean(deff) *180.0 / 3.141592653589793 

def angluar_error_nll_metric2(y_trues, y_preds):
   y_pred = y_preds[:, 0:3]
   
  
   #arccoss error
   deff =  tf.acos( K.clip(K.sum( tf.nn.l2_normalize(y_trues ,axis = -1)*tf.nn.l2_normalize(y_pred ,axis = -1) ,axis=-1,keepdims=True ) , 0.0, 1.))
   
   return  K.mean(deff)*180.0 / 3.141592653589793 


def error_estimate_mse(ytrue, ypreds):
    y_pred = ypreds[:, 0:3]
    sigma = ypreds[:, 3:] # + K.epsilon()
    sigma = tf.reshape(sigma,(-1,))
    errors =  tf.acos( K.clip(K.sum( tf.nn.l2_normalize(ytrue ,axis = -1)*tf.nn.l2_normalize(y_pred ,axis = -1) ,axis=-1,keepdims=True ) , 0.0, 1.))
    errors = tf.reshape(errors ,(-1,))
   
    #mse = K.sum( K.square(errors-sigma),axis=1)
    mse =  K.mean(tf.math.squared_difference(errors,sigma), axis=-1) # K.sum( tf(errors-sigma),axis=-1)



    return mse



# things i tried but didnt work out. 
def gaussian_nll_inv(ytrue, ypreds,lam=1):
    """Keras implmementation of multivariate Gaussian negative loglikelihood loss function. 
    This implementation implies diagonal covariance matrix.
    
    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)
        
    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples
        
    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam') 
    
    """

    mu = ypreds[:, 0:3]
    sigma = ypreds[:, 3:] # + K.epsilon()
    
    mse = K.sum( K.square(ytrue-mu)*sigma,axis=1)
    sigma_trace = K.sum(K.log(sigma ), axis=1)
    
    log_likelihood = mse - lam* sigma_trace

    return K.mean(log_likelihood)




def gaussian_nll_inv2(ytrue, ypreds):
    """Keras implmementation of multivariate Gaussian negative loglikelihood loss function. 
    This implementation implies diagonal covariance matrix.
    
    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)
        
    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples
        
    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam') 
    
    """

    mu = ypreds[:, 0:3]
    sigma = ypreds[:, 3:] # + K.epsilon()
    
  #  mse = K.sum( tf.square(ytrue-mu)*sigma),axis=1)
    mse = K.sum( tf.square(ytrue-mu)*tf.exp(sigma),axis=1)
   
 #   sigma_trace = K.sum(K.log(sigma +0.001 ), axis=1)
    sigma_trace = K.sum(tf.square(sigma) , axis=1)
    
    log_likelihood = mse -  sigma_trace

    return K.mean(log_likelihood)







def gaussian_nll_invlog(ytrue, ypreds):
    """Keras implmementation of multivariate Gaussian negative loglikelihood loss function. 
    This implementation implies diagonal covariance matrix.
    
    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)
        
    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples
        
    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam') 
    
    """

    mu = ypreds[:, 0:2]
    sigma = ypreds[:, 2:] # + K.epsilon()
    
    mse = K.sum( K.square(ytrue-mu)*sigma,axis=1)
    sigma_trace = K.sum(K.log(sigma ), axis=1)
    
    log_likelihood = mse -sigma_trace

    return K.mean(log_likelihood)
def gaussian_nll(ytrue, ypreds):
    """Keras implmementation of multivariate Gaussian negative loglikelihood loss function. 
    This implementation implies diagonal covariance matrix.
    
    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)
        
    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples
        
    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam') 
    
    """

    mu = ypreds[:, 0:3]
    sigma = ypreds[:, 3:] + K.epsilon()
    
    mse = K.sum( K.square(ytrue-mu)/sigma,axis=1)
   # sigma_trace = K.sum(K.log(sigma), axis=1)
    sigma_trace = K.sum(sigma, axis=1)
    log_likelihood = mse+sigma_trace

    return K.mean(log_likelihood)


