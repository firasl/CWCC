# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:20:33 2021

@author: laakom
"""

# import tensorflow as tf

# x = tf.Variable(tf.random.uniform([5, 10,10,3], -1, 1))

# s0, s1, s2 = tf.split(x, num_or_size_splits=3, axis=-1)
# c = tf.unstack(x,3,axis=-1) # (x, num_or_size_splits=3, axis=-1)




import os 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from utils_data_augmentation import augment_data
from tensorflow.keras.layers import Input,Dense, Lambda,BatchNormalization,Activation, multiply,Reshape,Flatten, Conv2D ,Lambda,Concatenate
from tensorflow.keras.layers import MaxPooling2D, Dropout,GlobalAveragePooling2D,concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
import numpy as np
#from parameters import *
from loss_utils import angluar_error_nll_metric2,error_estimate_mse
from costum_data_generator import DataGenerator_CWCCUU
from prediction_utilities import save_errors, ill_predict,print_angular_errors,angular_error_recovery,angular_error_reproduction
from INTEL_TAU_1080p import INTEL_TAU_DATASET
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, LambdaCallback,CSVLogger,LearningRateScheduler

from model_CWCC import CWCC







def get_aug_perfolder(sub_path):
    #read samples
    from glob import glob    
    aug_samples = glob(sub_path + "*.png")
    
    #get?groundtruth
    import pickle
    pkl_file = open( sub_path + 'ground_truth.pkl', 'rb')
    aug_labels = pickle.load(pkl_file)
    pkl_file.close()
    
    return aug_samples, aug_labels
def get_augmentedpartion(partition,ground_truths,aug_path):
    
   #train_data
  train_path =   aug_path + 'train/'
  aug_samples, aug_labels = get_aug_perfolder(train_path)  
  partition['train'] =  aug_samples 
  ground_truths['train'] =  aug_labels
   #validation_data
  val_path =   aug_path + 'val/'
  aug_samples, aug_labels = get_aug_perfolder(val_path)  
  partition['validation'] =  aug_samples 
  ground_truths['validation'] =  aug_labels

  return partition,ground_truths


def all_callbacks(path):
    
        
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss',
      		                                            patience=10,
      		                                            factor=0.5,
      		                                            min_lr=0.0000005)
    
    # stop training if no improvements are seen
    #early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=50,restore_best_weights=True)
    # saves model weights to file
    checkpoint=ModelCheckpoint(path , monitor='val_loss', save_best_only=True, save_weights_only=True)
    savecsvlog = CSVLogger(path[:-len('.hdf5')] + '_log.csv', separator=',', append=True )    
        
    def step_decay(epoch):
 	if epoch < 100:
           lrate = 0.0003
        elif epoch < 200:
            lrate = 0.00003
        else:
            lrate = 0.000003
        print('learning rate: ',lrate )
        return lrate
    lrschedule = LearningRateScheduler(step_decay)

    
    return  [learning_rate_reduction,savecsvlog,checkpoint,lrschedule] # ,[checkpoint,early_stop,lrschedule] 





def test_model(model,data,groundtruth,method,path,result_path):        
    model.load_weights( path + '.hdf5')
    print('predicting...') 
    gr, predictions = ill_predict(model,data,groundtruth,method)    
    old_rae = angular_error_recovery(gr, predictions)
    print('recovery errors : \n') 
    print_angular_errors( old_rae)
    save_errors(old_rae,result_path + 'recovery_error')
    new_err = angular_error_reproduction(gr, predictions)
    print('reproduction errors : \n') 
    print_angular_errors(new_err)
    save_errors(new_err,result_path+'reproduction_error')
    return  gr, predictions 




if __name__ == '__main__':
   

    dataset_params = {'camera_names':['Canon_5DSR','Nikon_D810','Sony_IMX135_BLCCSC'],
                      'set_names': ['field_3_cameras','field_1_cameras','lab_printouts','lab_realscene'],
                      'root_path': '/mnt/Firas/intel_tau/processed_1080p'
                          }
    #dataset_object
    inteltau = INTEL_TAU_DATASET(**dataset_params)
    inteltau.set_subsets_splits()   
    input_shape= (227,227)
     
    method = 'CWCCUU_squuezenet'
    EPOCHS = 350  
    phaseone = True
	#phase 2 is for the uncertainty estimation block. make it True on case you want to train it. 
    twoEPOCHS = 50
    two_phases = False

    if not os.path.isdir('trained_models'):
         try: 
             os.mkdir('trained_models')
         except:
             print("An exception occurred while creating folder", 'trained_models')
 
             
    if not os.path.isdir('trained_models//'+ method):
         try: 
             os.mkdir('trained_models//' + method)
         except:
             print("An exception occurred while creating folder", 'trained_models//'+ method) 
    
 
    
 
    if not os.path.isdir('errors'):
         try: 
             os.mkdir('errors')
         except:
             print("An exception occurred while creating folder", 'errors')
             
             
    if not os.path.isdir('errors//'+ method):
         try: 
             os.mkdir('errors//'+ method)
         except:
             print("An exception occurred while creating folder", 'errors' + method)





             
    #for method in methods:
    for fold in range(0,1):
            print(fold)
           #     train_cam =    'Nikon_D810'     
       
            partition,ground_truths = inteltau.get_train__test_10folds(fold)
                
            # Dataset partition: train-validation-test
                        
            path =  'trained_models//'  + method + '//model_fold'  + str(fold)  
            aug_path  = dataset_params['root_path'] + '/expirement_227*227_fold_' + str(fold) + '/'
            
            if os.path.isdir(aug_path):
               print( 'augmented dataset found in ',aug_path ) 
            else:
               print('creating augmented data for fold ', fold) 
               try: 
                   os.mkdir(aug_path)
               except:
                   print("An exception occurred while creating ", aug_path) 
               
               #training data_augmentation:
               train_dir = aug_path + '/train'
               try: 
                   os.mkdir(train_dir)
               except:
                   print("An exception occurred while creating ", train_dir)                
               augment_data(20*len(partition['train']),partition['train'],ground_truths['train'],input_shape,train_dir)    
               #validation data_augmentation:
               val_dir = aug_path + '/val'
               try: 
                   os.mkdir(val_dir)
               except:
                   print("An exception occurred while creating ", val_dir)                
               augment_data(5*len(partition['validation']),partition['validation'],ground_truths['validation'],input_shape,val_dir)  
            
            partition,ground_truths = get_augmentedpartion(partition,ground_truths,aug_path)
            

            model = CWCC(input_shape= input_shape)
            
            for layer in model.layers:
                if layer.name[0:3]== 'var':
                    layer.trainable = False
                    print(layer.name)
            
            model.summary()



            if os.path.isfile(path + '.hdf5'):
                 print('Warning Previous model found. \n  loading weights....')
                 model.load_weights(path + '.hdf5')  
                 # Parameters
            train_params = {'dim': input_shape,
                   'batch_size': 16,
                   'n_channels': 3,
                   'shuffle': True}
            val_params = {'dim': input_shape,
                   'batch_size': 16,
                   'n_channels': 3,
                   'shuffle': True}
            
            print('creating data generators for training with fold', fold )
            training_generator = DataGenerator_CWCCUU(partition['train'], ground_truths['train'], **train_params)
            validation_generator = DataGenerator_CWCCUU(partition['validation'], ground_truths['validation'], **val_params)
                                                       
                      
      
            if phaseone == True:
                print(method +' training with fold  ', fold )
                history = model.fit(generator=training_generator, epochs=EPOCHS,
                                validation_data=validation_generator,
                                steps_per_epoch = (len(partition['train']) // train_params['batch_size']) ,                    
                                use_multiprocessing=True, 
                                callbacks =all_callbacks( path + '.hdf5' ),
                                workers=16)
                
                
                    
                result_path =  'errors//' + method + '//model_phase1_fold' + str(fold)  
                model.load_weights(path + '.hdf5') 
                test_model(model,partition['test'],ground_truths['test'],method + '227',path,result_path)
            if two_phases == True :                 
                model.load_weights(path + '.hdf5') 
                path2 = path  + '_phase2.hdf5'
                for layer in model.layers:
                        layer.trainable = False
                        print('phase2' + layer.name)
                            
                for layer in model.layers:
                    if layer.name[0:3]== 'var':
                        layer.trainable = True
                        print(layer.name)
                model.compile(loss=error_estimate_mse,metrics=[angluar_error_nll_metric2], optimizer=Adam(lr = 0.03))
                #checkpoint2=ModelCheckpoint(path2 , monitor='val_loss', save_best_only=True, save_weights_only=True)
                savecsvlog2 = CSVLogger(path2[:-len('.hdf5')] + '_log.csv', separator=',', append=True ) 
                model.save_weights(path2)                
                history = model.fit(generator=training_generator, epochs=twoEPOCHS,
                                validation_data=validation_generator,
                                steps_per_epoch = (len(partition['train']) // train_params['batch_size']) ,                    
                                use_multiprocessing=True, 
                                callbacks =[savecsvlog2],
                                workers=16)
                model.save_weights(path2)                
                
                result_path =  'errors//' + method + '//model_phase2_fold' + str(fold)  
                model.load_weights(path2) 
                test_model(model,partition['test'],ground_truths['test'],method ,path2[:-len('.hdf5')],result_path)



























