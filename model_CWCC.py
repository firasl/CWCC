# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:05:59 2021

@author: laakom
"""


from tensorflow.keras.layers import Input,Dense, Lambda,BatchNormalization,Activation, multiply,Reshape,Flatten, Conv2D ,Lambda,Concatenate
from tensorflow.keras.layers import MaxPooling2D, Dropout,GlobalAveragePooling2D,concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
import numpy as np
#from parameters import *
from loss_utils import angluar_error_nll2,angluar_error_nll_metric2
from tensorflow.keras.regularizers import l2

from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


def fire_module(x, fire_id, squeeze=16, expand=64):
   s_id = 'fire' + str(fire_id) + '/'
   x = Conv2D(squeeze, (1, 1),activation='relu', padding='valid', kernel_initializer='he_normal',name=s_id + '_squeeze')(x)

   left = Conv2D(expand, (1, 1),activation='relu', padding='valid',kernel_initializer='he_normal', name=s_id + '_expand_1x1')(x)

   right = Conv2D(expand, (3, 3),activation='relu', padding='same',kernel_initializer='he_normal', name=s_id + '_expand_3x3',kernel_regularizer=l2(1e-4))(x)

   x = concatenate([left, right], axis=3, name=s_id + 'concat')
   return x

def chanel_submodel(input_shape= (227,227,1),show_summary= False) :

    X_input = Input(input_shape)
#    # Stage 1
    X = Conv2D(64, (3, 3),strides=(2, 2), activation='relu',kernel_initializer='glorot_uniform',padding ='same',kernel_regularizer=l2(1e-4))(X_input)
    X = MaxPooling2D((3,3),strides=(2, 2))(X)

    X = fire_module(X, fire_id=2, squeeze=16, expand=64)
    X = fire_module(X, fire_id=3, squeeze=16, expand=64)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(X)
    
    X = fire_module(X, fire_id=4, squeeze=32, expand=128)
    Y = fire_module(X, fire_id=5, squeeze=32, expand=128)
    
    
    Y = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), name='pool5')(X)    



    chanel_model = Model(inputs = X_input, outputs = Y)
       
    return chanel_model



def CWCC(input_shape= (227,227,3), show_summary= False) :
    
    X_input = Input(input_shape)
    # we split the input channel-wise::
    X_inputb, X_inputg, X_inputr = tf.split(X_input, num_or_size_splits=3, axis=-1) 

    #we create the channel-model--
    sub_model = chanel_submodel()
    if  show_summary:
        sub_model.summary()
    
    # passing each channel through our model..
    ped_r = sub_model(X_inputr)
    ped_g= sub_model(X_inputg)   
    ped_b= sub_model(X_inputb)     
    
    # concatinating the output maps
    mixed_channels = concatenate([ped_r, ped_g,ped_b], axis=-1, name='concatinated_channels')
    
    mixed_channels = GlobalAveragePooling2D()(mixed_channels)  
    mixed_channels = Dropout(rate = 0.1)(mixed_channels)
    total_pred = Dense(40, activation='relu', name='mixed_features',kernel_regularizer=l2(1e-4))(mixed_channels)    
    total_pred = Dropout(rate = 0.1)(total_pred)
    estimation = Dense(3,activation='relu')(total_pred)
    estimation = Lambda(lambda x: K.l2_normalize(x), name='estimation')(estimation);

    var = Dense(3,activation='relu',name='variance')(total_pred)

    total_output = Concatenate()([estimation,var])   

    final_model = Model(inputs = X_input , outputs = total_output, name='CWCCUU_squuezemodel');

    if show_summary:
        final_model.summary()
        
    final_model.compile(loss=angluar_error_nll2,metrics=[angluar_error_nll_metric2], optimizer=Adam(lr = 0.0005))    
   
    return final_model
