# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 17:45:22 2022

@author: marcel.rotunno
"""

from osgeo import gdal, gdal_array
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os, shutil
import math
import tensorflow as tf
import pickle
import random
import time
import datetime
import types
import gc
from pathlib import Path

from tensorflow.keras.layers import Input, concatenate, Conv2D, BatchNormalization, Activation 
from tensorflow.keras.layers import MaxPool2D, Conv2DTranspose, UpSampling2D, Concatenate
from tensorflow.keras.layers import Dropout, Add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Metric
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras import backend as K

from unetr_2d import config_dict, build_unetr_2d, Patches
print(config_dict)
from segformer_tf_k2.models import SegFormer_B0
from segformer_tf_k2.models import SegFormer_B1
from segformer_tf_k2.models import SegFormer_B2
from segformer_tf_k2.models import SegFormer_B3
from segformer_tf_k2.models import SegFormer_B4
from segformer_tf_k2.models import SegFormer_B5
from segformer_tf_k2.models.modules import MixVisionTransformer
from segformer_tf_k2.models.Head import SegFormerHead
from segformer_tf_k2.models.utils import ResizeLayer


from sklearn.utils import shuffle

from sklearn.metrics import (confusion_matrix, f1_score, precision_score, 
                            recall_score, accuracy_score, ConfusionMatrixDisplay)

from scipy.ndimage import gaussian_filter

from tensorflow.keras.optimizers import Adam, SGD
#from focal_loss import SparseCategoricalFocalLoss



import matplotlib.pyplot as plt

def salva_arrays(folder, **kwargs):
    '''
    

    Parameters
    ----------
    folder : str
        The string of the folder to save the files.
    **kwargs : keywords and values
        Keyword (name of the file) and the respective value (numpy array)

    Returns
    -------
    None.

    '''
    for kwarg in kwargs:
        if not os.path.exists(os.path.join(folder, kwarg) + '.npy'):
            np.save(os.path.join(folder, kwarg) + '.npy', kwargs[kwarg])
        else:
            print(kwarg, 'não foi salvo pois já existe')





# Res-UNet

def batchnorm_relu(inputs):
    """ Batch Normalization & ReLU """
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    return x


def batchnorm_elu(inputs):
    """ Batch Normalization & ReLU """
    x = BatchNormalization()(inputs)
    x = Activation("elu")(x)
    
    return x

def residual_block(inputs, num_filters, strides=1):
    """ Convolutional Layers """
    x = batchnorm_relu(inputs)
    x = Conv2D(num_filters, 3, padding="same", strides=strides)(x)
    x = batchnorm_relu(x)
    x = Conv2D(num_filters, 3, padding="same", strides=1)(x)
    """ Shortcut Connection (Identity Mapping) """
    s = Conv2D(num_filters, 1, padding="same", strides=strides)(inputs)
    """ Addition """
    x = x + s
    return x

def decoder_block(inputs, skip_features, num_filters):
    """ Decoder Block """
    x = UpSampling2D((2, 2))(inputs)
    x = Concatenate()([x, skip_features])
    x = residual_block(x, num_filters, strides=1)
    return x

def build_model_resunet(input_shape, n_classes):
    """ RESUNET Architecture """
    inputs = Input(input_shape)
    """ Endoder 1 """
    x = Conv2D(64, 3, padding="same", strides=1)(inputs)
    x = batchnorm_relu(x)
    x = Conv2D(64, 3, padding="same", strides=1)(x)
    s = Conv2D(64, 1, padding="same")(inputs)
    s1 = x + s
    """ Encoder 2, 3 """
    s2 = residual_block(s1, 128, strides=2)
    s3 = residual_block(s2, 256, strides=2)
    """ Bridge """
    b = residual_block(s3, 512, strides=2)

    """ Decoder 1, 2, 3 """
    x = decoder_block(b, s3, 256)
    x = decoder_block(x, s2, 128)
    x = decoder_block(x, s1, 64)
    """ Classifier """
    outputs = Conv2D(n_classes, 1, padding="same", activation="softmax")(x)
    """ Model """
    model = Model(inputs, outputs, name="Res-UNet")
    return model


def build_model_unet(input_shape, n_classes):
    # U-Net architecture
    input_img = Input(input_shape)

    # Contract stage
    f1 = 64
    b1conv1 = Conv2D(f1 , (3 , 3) , activation='relu' , padding='same', name = 'b1conv1')(input_img)
    b1conv2 = Conv2D(f1 , (3 , 3) , activation='relu' , padding='same', name = 'b1conv2')(b1conv1)

    pool1 = MaxPool2D((2 , 2), name = 'pooling1')(b1conv2)

    b2conv1 = Conv2D(f1*2 , (3 , 3) , activation='relu' , padding='same', name = 'b2conv1')(pool1)
    b2conv2 = Conv2D(f1*2 , (3 , 3) , activation='relu' , padding='same', name = 'b2conv2')(b2conv1)

    pool2 = MaxPool2D((2 , 2), name = 'pooling2')(b2conv2)

    b3conv1 = Conv2D(f1*4 , (3 , 3) , activation='relu' , padding='same', name = 'b3conv1')(pool2)
    b3conv2 = Conv2D(f1*4 , (3 , 3) , activation='relu' , padding='same', name = 'b3conv2')(b3conv1)

    pool3 = MaxPool2D((2 , 2), name = 'pooling3')(b3conv2)

    b4conv1 = Conv2D(f1*8 , (3 , 3) , activation='relu' , padding='same', name = 'b4conv1')(pool3)
    b4conv2 = Conv2D(f1*8 , (3 , 3) , activation='relu' , padding='same', name = 'b4conv2')(b4conv1)

    pool4 = MaxPool2D((2 , 2), name = 'pooling4')(b4conv2)

    b5conv1 = Conv2D(f1*16 , (3 , 3) , activation='relu' , padding='same', name = 'b5conv1')(pool4)
    b5conv2 = Conv2D(f1*16 , (3 , 3) , activation='relu' , padding='same', name = 'b5conv2')(b5conv1)

    # Expansion stage
    upsample1 = Conv2DTranspose(f1*8, (3 , 3), strides=(2,2), activation = 'relu', padding = 'same', name = 'upsampling1')(b5conv2)
    concat1 = concatenate( [upsample1,b4conv2] )
    b6conv1 = Conv2D(f1*8 , (3 , 3) , activation='relu' , padding='same', name = 'b6conv1')(concat1)
    b6conv2 = Conv2D(f1*8 , (3 , 3) , activation='relu' , padding='same', name = 'b6conv2')(b6conv1)
    
    upsample2 = Conv2DTranspose(f1*4, (3 , 3), strides=(2,2), activation = 'relu', padding = 'same', name = 'upsampling2')(b6conv2)
    concat2 = concatenate( [upsample2,b3conv2] )
    b7conv1 = Conv2D(f1*4 , (3 , 3) , activation='relu' , padding='same', name = 'b7conv1')(concat2)
    b7conv2 = Conv2D(f1*4 , (3 , 3) , activation='relu' , padding='same', name = 'b7conv2')(b7conv1)

    upsample3 = Conv2DTranspose(f1*2, (3 , 3), strides=(2,2), activation = 'relu', padding = 'same', name = 'upsampling3')(b7conv2)
    concat3 = concatenate( [upsample3,b2conv2] )
    b8conv1 = Conv2D(f1*2 , (3 , 3) , activation='relu' , padding='same', name = 'b8conv1')(concat3)
    b8conv2 = Conv2D(f1*2 , (3 , 3) , activation='relu' , padding='same', name = 'b8conv2')(b8conv1)

    upsample4 = Conv2DTranspose(f1, (3 , 3), strides=(2,2), activation = 'relu', padding = 'same', name = 'upsampling4')(b8conv2)
    concat4 = concatenate( [upsample4,b1conv2] )
    b9conv1 = Conv2D(f1 , (3 , 3) , activation='relu' , padding='same', name = 'b9conv1')(concat4)
    b9conv2 = Conv2D(f1 , (3 , 3) , activation='relu' , padding='same', name = 'b9conv2')(b9conv1)

    # Output segmentation
    b9conv3 = Conv2D(f1 , (3 , 3) , activation='relu' , padding='same', name = 'b9conv3')(b9conv2)
    b9conv4 = Conv2D(f1 , (3 , 3) , activation='relu' , padding='same', name = 'b9conv4')(b9conv3)

    output = Conv2D(n_classes,(1,1), activation = 'softmax')(b9conv4)


    return Model(inputs = input_img, outputs = output, name='U-Net')



def resnet_block_chamorro(x, n_filter, ind, dropout=True, dropout_rate=0.2):
    x_init = x

    ## Conv 1
    x = Conv2D(n_filter, (3, 3), activation='relu', padding="same", name = 'res1_net'+str(ind))(x)
    if dropout:
        x = Dropout(dropout_rate, name = 'drop_net'+str(ind))(x)
        print(f'ATENÇÃO DROPOUT RATE = {dropout_rate}')
    else:
        pass
    ## Conv 2
    x = Conv2D(n_filter, (3, 3), activation='relu', padding="same", name = 'res2_net'+str(ind))(x)
    
    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), activation='relu', padding="same", name = 'res3_net'+str(ind))(x_init)
    
    ## Add
    x = Add()([x, s])
    return x






# Residual U-Net model
def build_resunet_chamorro(input_shape, nb_filters, n_classes, last_activation='softmax'):
    '''Base network to be shared (eq. to feature extraction)'''
    input_layer= Input(shape = input_shape, name="input_enc_net")
    
    res_block1 = resnet_block_chamorro(input_layer, nb_filters[0], 1)
    pool1 = MaxPool2D((2 , 2), name='pool_net1')(res_block1)
    
    res_block2 = resnet_block_chamorro(pool1, nb_filters[1], 2) 
    pool2 = MaxPool2D((2 , 2), name='pool_net2')(res_block2)
    
    res_block3 = resnet_block_chamorro(pool2, nb_filters[2], 3) 
    pool3 = MaxPool2D((2 , 2), name='pool_net3')(res_block3)
    
    res_block4 = resnet_block_chamorro(pool3, nb_filters[2], 4)
    
    res_block5 = resnet_block_chamorro(res_block4, nb_filters[2], 5)
    
    res_block6 = resnet_block_chamorro(res_block5, nb_filters[2], 6)
    
    upsample3 = Conv2D(nb_filters[2], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net3')(UpSampling2D(size = (2,2))(res_block6))
    
    merged3 = concatenate([res_block3, upsample3], name='concatenate3')

    upsample2 = Conv2D(nb_filters[1], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net2')(UpSampling2D(size = (2,2))(merged3))
                                                 
    merged2 = concatenate([res_block2, upsample2], name='concatenate2')
                                                                                          
    upsample1 = Conv2D(nb_filters[0], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net1')(UpSampling2D(size = (2,2))(merged2))
    merged1 = concatenate([res_block1, upsample1], name='concatenate1')

    output = Conv2D(n_classes,(1,1), activation = last_activation, padding = 'same', name = 'output')(merged1)
                                                                                                           
    return Model(input_layer, output)

"""
# Residual U-Net model para Pós-Processamento
def build_resunet_chamorro_semdropout(input_shape, nb_filters, n_classes, last_activation='softmax'):
    '''Base network to be shared (eq. to feature extraction)'''
    input_layer= Input(shape = input_shape, name="input_enc_net")
    
    res_block1 = resnet_block_chamorro(input_layer, nb_filters[0], 1, dropout=False)
    pool1 = MaxPool2D((2 , 2), name='pool_net1')(res_block1)
    
    res_block2 = resnet_block_chamorro(pool1, nb_filters[1], 2, dropout=False) 
    pool2 = MaxPool2D((2 , 2), name='pool_net2')(res_block2)
    
    upsample2 = Conv2D(nb_filters[1], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net2')(UpSampling2D(size = (2,2))(pool2))
                                                 
    merged2 = concatenate([res_block2, upsample2], name='concatenate2')
                                                                                          
    upsample1 = Conv2D(nb_filters[0], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net1')(UpSampling2D(size = (2,2))(merged2))
    merged1 = concatenate([res_block1, upsample1], name='concatenate1')

    output = Conv2D(n_classes,(1,1), activation = last_activation, padding = 'same', name = 'output')(merged1)
                                                                                                           
    return Model(input_layer, output)
"""


# Residual U-Net model para Pós-Processamento
def build_resunet_chamorro_semdropout_curta(input_shape, nb_filters, n_classes, last_activation='softmax'):
    '''Base network to be shared (eq. to feature extraction)'''
    print('Rede curta')
    
    input_layer= Input(shape = input_shape, name="input_enc_net")
    
    res_block1 = resnet_block_chamorro(input_layer, nb_filters[0], 1, dropout=False)
    pool1 = MaxPool2D((2 , 2), name='pool_net1')(res_block1)
    
    res_block2 = resnet_block_chamorro(pool1, nb_filters[1], 2, dropout=False) 
    pool2 = MaxPool2D((2 , 2), name='pool_net2')(res_block2)
    
    res_block3 = resnet_block_chamorro(pool2, nb_filters[2], 3, dropout=False)
    
    upsample2 = Conv2D(nb_filters[1], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net2')(UpSampling2D(size = (2,2))(res_block3))
                                                 
    merged2 = concatenate([res_block2, upsample2], name='concatenate2')
                                                                                          
    upsample1 = Conv2D(nb_filters[0], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net1')(UpSampling2D(size = (2,2))(merged2))
    merged1 = concatenate([res_block1, upsample1], name='concatenate1')

    output = Conv2D(n_classes,(1,1), activation = last_activation, padding = 'same', name = 'output')(merged1)
                                                                                                           
    return Model(input_layer, output)


# Residual U-Net model para Pós-Processamento
def build_resunet_chamorro_semdropout(input_shape, nb_filters, n_classes, last_activation='softmax'):
    '''Base network to be shared (eq. to feature extraction)'''
    input_layer= Input(shape = input_shape, name="input_enc_net")
    
    res_block1 = resnet_block_chamorro(input_layer, nb_filters[0], 1, dropout=False)
    pool1 = MaxPool2D((2 , 2), name='pool_net1')(res_block1)
    
    res_block2 = resnet_block_chamorro(pool1, nb_filters[1], 2, dropout=False) 
    pool2 = MaxPool2D((2 , 2), name='pool_net2')(res_block2)
    
    res_block3 = resnet_block_chamorro(pool2, nb_filters[2], 3, dropout=False) 
    pool3 = MaxPool2D((2 , 2), name='pool_net3')(res_block3)
    
    res_block4 = resnet_block_chamorro(pool3, nb_filters[2], 4, dropout=False)
    
    res_block5 = resnet_block_chamorro(res_block4, nb_filters[2], 5, dropout=False)
    
    res_block6 = resnet_block_chamorro(res_block5, nb_filters[2], 6, dropout=False)
    
    upsample3 = Conv2D(nb_filters[2], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net3')(UpSampling2D(size = (2,2))(res_block6))
    
    merged3 = concatenate([res_block3, upsample3], name='concatenate3')

    upsample2 = Conv2D(nb_filters[1], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net2')(UpSampling2D(size = (2,2))(merged3))
                                                 
    merged2 = concatenate([res_block2, upsample2], name='concatenate2')
                                                                                          
    upsample1 = Conv2D(nb_filters[0], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net1')(UpSampling2D(size = (2,2))(merged2))
    merged1 = concatenate([res_block1, upsample1], name='concatenate1')

    output = Conv2D(n_classes,(1,1), activation = last_activation, padding = 'same', name = 'output')(merged1)
                                                                                                           
    return Model(input_layer, output)



# Model CorrED
def conv_block_corred(inputs, num_filters, part='encoder'):
    """ Convolutional Layers """
    if part=='encoder':
        x = Conv2D(num_filters , (3 , 3) , padding='same', strides=2)(inputs)
    elif part=='decoder':
        x = Conv2D(num_filters , (3 , 3) , padding='same', strides=1)(inputs)
    else:
        raise Exception('Valor de parte deve "encoder" ou "decoder"')
        
    x = batchnorm_elu(x)
        
    x = Conv2D(num_filters , (3 , 3) , padding='same', strides=1)(x)
    
    x = batchnorm_elu(x)
    
    return x

def build_model_corred(input_shape, n_classes):
    input_img = Input(input_shape)
    
    '''Encoder'''
    # Bloco Encoder 1 - 2 Convoluções seguindas de um Batch Normalization com Relu
    b1 = conv_block_corred(input_img, 16)
    
    # Bloco Encoder 2 - " " "
    b2 = conv_block_corred(b1, 32, part='encoder')
    
    # Bloco Encoder 3
    b3 = conv_block_corred(b2, 32, part='encoder')
    
    # Bloco Encoder 4
    b4 = conv_block_corred(b3, 32, part='encoder')
    
    
    '''Decoder'''
    # Bloco Decoder 1 - Upsample seguido de 2 convoluções
    up1 = UpSampling2D((2, 2), interpolation='bilinear')(b4)
    up1 = conv_block_corred(up1, 16, part='decoder')
    
    # Bloco Decoder 2 - " " "
    up2 = UpSampling2D((2, 2), interpolation='bilinear')(up1)
    up2 = conv_block_corred(up2, 16, part='decoder')
    
    # Bloco Decoder 3
    up3 = UpSampling2D((2, 2), interpolation='bilinear')(up2)
    up3 = conv_block_corred(up3, 16, part='decoder')
    
    # Bloco Decoder 4
    up4 = UpSampling2D((2, 2), interpolation='bilinear')(up3)
    up4 = conv_block_corred(up4, 16, part='decoder')
    
    # Final Convolution
    output = Conv2D(n_classes, (1,1), activation = 'softmax')(up4)
    
    
    return Model(inputs = input_img, outputs = output, name='CorrED')

        
# Build UNet or Res-UNet or CorrED
def build_model(input_shape, n_classes, model_type='unet'):
    if model_type == 'unet':
        return build_model_unet(input_shape, n_classes)
    
    elif model_type == 'resunet':
        return build_model_resunet(input_shape, n_classes)
    
    elif model_type == 'resunet chamorro':
        return build_resunet_chamorro(input_shape, (64, 128, 256), n_classes)
    
    elif model_type == 'corred':
        return build_model_corred(input_shape, n_classes)

    elif model_type == 'resunet chamorro pos':
         return build_resunet_chamorro_semdropout(input_shape, (64, 128, 256), n_classes)
     
    elif model_type == 'resunet chamorro pos curta':
         return build_resunet_chamorro_semdropout_curta(input_shape, (64, 128, 256), n_classes)
     
    elif model_type == 'unet transformer':
        return build_unetr_2d(input_shape, config_dict)
    
    elif model_type == 'segformer_b0':
        return SegFormer_B0(input_shape, n_classes)

    elif model_type == 'segformer_b1':
        return SegFormer_B1(input_shape, n_classes)  

    elif model_type == 'segformer_b2':
        return SegFormer_B2(input_shape, n_classes)    
    
    elif model_type == 'segformer_b3':
        return SegFormer_B3(input_shape, n_classes)  
    
    elif model_type == 'segformer_b4':
        return SegFormer_B4(input_shape, n_classes)  
    
    elif model_type == 'segformer_b5':
        return SegFormer_B5(input_shape, n_classes)  
    
    else:
        raise Exception("Model options are 'unet' and 'resunet' and 'resunet chamorro' and 'corred' and "
                        "'resunet chamorro pos' and 'resunet chamorro pos curta' and 'unet transformer'")
        




    


def train_unet(net, x_train, y_train, x_valid, y_valid, batch_size, epochs, early_stopping_epochs, early_stopping_delta, filepath, 
               filename, early_stopping=True, early_loss=False, metric_name='f1score', lr_decay=False):
    print('Start the training...')
    
    early_stop = None
    filepath_name = os.path.join(filepath, filename+'.h5')
    log_dir = os.path.join(filepath, 'logs', 'fit') #, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    shutil.rmtree(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    
    # Com Learning Rate Decay
    if lr_decay:
        initial_lrate = 0.001
        drop = 0.1
        epochs_drop = 30.0
        print(f'Initial Learning Rate={initial_lrate}, Drop={drop}, Epochs Drop={epochs_drop}')        
        def step_decay(epoch):
            lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
            return lrate
        
        lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)

    
    # Com Early Stopping
    if early_stopping:
        # Early Stopping por Loss
        if early_loss:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_epochs, mode='min', restore_best_weights=True,
                                                          min_delta=early_stopping_delta, verbose=1)
            
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath_name,
                                                              monitor='val_loss',
                                                              mode='min',
                                                              save_weights_only=False,
                                                              verbose=1,
                                                              save_freq='epoch',
                                                              save_best_only=True)
            
        # Early Stopping por Métrica
        else:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_'+metric_name, patience=early_stopping_epochs, mode='max', restore_best_weights=True,
                                                          min_delta=early_stopping_delta, verbose=1)
            
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath_name,
                                                              monitor='val_'+metric_name,
                                                              mode='max',
                                                              save_weights_only=False,
                                                              verbose=1,
                                                              save_freq='epoch',
                                                              save_best_only=True)
    
    # Sem Early Stopping salva apenas o último modelo
    else:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath_name, 
                                                         monitor='val_loss',
                                                         mode='auto',
                                                         save_weights_only=False,
                                                         verbose=1,
                                                         save_freq='epoch',
                                                         save_best_only=False)
    
    # Constroi lista de callbacks
    if lr_decay:
        if early_stop:
            callbacks = [early_stop, cp_callback, tensorboard_callback, lrate]
        else:
            callbacks = [cp_callback, tensorboard_callback, lrate]
    
    else:
        if early_stop:
            callbacks = [early_stop, cp_callback, tensorboard_callback]
        else:
            callbacks = [cp_callback, tensorboard_callback]
        
        
    # Treina Modelo
    history = net.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose='auto',
                      callbacks=callbacks, validation_data=(x_valid, y_valid))
    
    # Retorna resultados de Treino e Validação
    historia = history.history
    
    lista_loss = historia['loss']
    lista_metric = historia[metric_name]
    lista_val_loss = historia['val_loss']
    lista_val_metric = historia['val_' + metric_name]
    
    history_train = [np.array([dupla]) for dupla in zip(lista_loss, lista_metric)]
    history_valid = [np.array([dupla]) for dupla in zip(lista_val_loss, lista_val_metric)]
    resultado = [ history_train, history_valid ]
    
    return resultado


def compute_relaxed_metrics(y: np.ndarray, pred: np.ndarray, buffer_y: np.ndarray, buffer_pred: np.ndarray,
                            nome_conjunto: str, print_file=None):
    '''
    

    Parameters
    ----------
    y : np.ndarray
        array do formato (batches, heigth, width, channels).
    pred : np.ndarray
        array do formato (batches, heigth, width, channels).
    buffer_y : np.ndarray
        array do formato (batches, heigth, width, channels).
    buffer_pred : np.ndarray
        array do formato (batches, heigth, width, channels).

    Returns
    -------
    relaxed_precision : float
        Relaxed Precision (by 3 pixels).
    relaxed_recall : float
        Relaxed Recall (by 3 pixels).
    relaxed_f1score : float
        Relaxed F1-Score (by 3 pixels).

    '''
    
    
    # Flatten arrays to evaluate quality
    true_labels = np.reshape(y, (y.shape[0] * y.shape[1] * y.shape[2]))
    predicted_labels = np.reshape(pred, (pred.shape[0] * pred.shape[1] * pred.shape[2]))
    
    buffered_true_labels = np.reshape(buffer_y, (buffer_y.shape[0] * buffer_y.shape[1] * buffer_y.shape[2]))
    buffered_predicted_labels = np.reshape(buffer_pred, (buffer_pred.shape[0] * buffer_pred.shape[1] * buffer_pred.shape[2]))
    
    # Calculate Relaxed Precision and Recall for test data
    relaxed_precision = 100*precision_score(buffered_true_labels, predicted_labels, pos_label=1)
    relaxed_recall = 100*recall_score(true_labels, buffered_predicted_labels, pos_label=1)
    relaxed_f1score = 2  *  (relaxed_precision*relaxed_recall) / (relaxed_precision+relaxed_recall)
    
    # Print result
    if print_file:
        print('\nRelaxed Metrics for %s' % (nome_conjunto), file=print_file)
        print('=======', file=print_file)
        print('Relaxed Precision: ', relaxed_precision, file=print_file)
        print('Relaxed Recall: ', relaxed_recall, file=print_file)
        print('Relaxed F1-Score: ', relaxed_f1score, file=print_file)
        print()        
    else:
        print('\nRelaxed Metrics for %s' % (nome_conjunto))
        print('=======')
        print('Relaxed Precision: ', relaxed_precision)
        print('Relaxed Recall: ', relaxed_recall)
        print('Relaxed F1-Score: ', relaxed_f1score)
        print()
    
    return relaxed_precision, relaxed_recall, relaxed_f1score


# Função que mostra gráfico
def show_graph_loss_accuracy(history, accuracy_position, metric_name = 'accuracy', save=False, save_path=r'', save_name='plotagem.png'):
    plt.rcParams['axes.facecolor']='white'
    plt.figure(num=1, figsize=(14,6))

    config = [ { 'title': 'model %s' % (metric_name), 'ylabel': '%s' % (metric_name), 'legend_position': 'upper left', 'index_position': accuracy_position },
               { 'title': 'model loss', 'ylabel': 'loss', 'legend_position': 'upper right', 'index_position': 0 } ]

    for i in range(len(config)):
        
        plot_number = 120 + (i+1)
        plt.subplot(plot_number)
        plt.plot(history[0,:,0, config[i]['index_position']])
        plt.plot(history[1,:,0, config[i]['index_position']])
        plt.title(config[i]['title'])
        plt.ylabel(config[i]['ylabel'])
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc=config[i]['legend_position'])
        plt.tight_layout()
        
    if save:
        plt.savefig(save_path + save_name)
        plt.close()
    else:
        plt.show(block=False)
    

    
    

# Calcula os limites em x e y do patch, para ser usado no caso de um patch de borda
def calculate_xp_yp_limits(p_size, x, y, xmax, ymax, left_half, up_half, right_shift, down_shift):
    # y == 0
    if y == 0:
        if x == 0:
            if y + p_size >= ymax:
                yp_limit = ymax - y 
            else:
                yp_limit = p_size
                
            if x + p_size >= xmax:
                xp_limit = xmax - x
            else:
                xp_limit = p_size
                
        else:
            if y + p_size >= ymax:
                yp_limit = ymax - y
            else:
                yp_limit = p_size
                
            if x + right_shift >= xmax:
                xp_limit = xmax - x + left_half
            else:
                xp_limit = p_size
    # y != 0
    else:
        if x == 0:
            if y + down_shift >= ymax:
                yp_limit = ymax - y + up_half
            else:
                yp_limit = p_size
                
            if x + p_size >= xmax:
                xp_limit = xmax - x
            else:
                xp_limit = p_size
        
        else:
            if y + down_shift >= ymax:
                yp_limit = ymax - y + up_half
            else:
                yp_limit = p_size
                
            if x + right_shift >= xmax:
                xp_limit = xmax - x + left_half
            else:
                xp_limit = p_size

    return xp_limit, yp_limit

# Cria o mosaico a partir dos batches extraídos da Imagem de Teste
def unpatch_reference(reference_batches, stride, reference_shape, border_patches=False):
    '''
    Function: unpatch_reference
    -------------------------
    Unpatch the patches of a reference to form a mosaic
    
    Input parameters:
      reference_batches  = array containing the batches (batches, h, w)
      stride = stride used to extract the patches
      reference_shape = shape of the target mosaic. Is the same shape from the labels image. (h, w)
      border_patches = include patches overlaping image borders, as extracted with the function extract_patches
    
    Returns: 
      mosaic = array in shape (h, w) with the mosaic build from the patches.
               Array is builded so that, in overlap part, half of the patch is from the left patch,
               and the other half is from the right patch. This is done to lessen border effects.
               This happens also with the patches in the vertical.
    '''
    
    # Objetivo é reconstruir a imagem de forma que, na parte da sobreposição dos patches,
    # metade fique a cargo do patch da esquerda e a outra metade, a cargo do patch da direita.
    # Isso é para diminuir o efeito das bordas

    # Trabalhando com patch quadrado (reference_batches.shape[1] == reference_batches.shape[2])
    # ovelap e notoverlap são o comprimento de pixels dos patches que tem ou não sobreposição 
    # Isso depende do stride com o qual os patches foram extraídos
    p_size = reference_batches.shape[1]
    overlap = p_size - stride
    notoverlap = stride
    
    # Cálculo de metade da sobreposição. Cada patch receberá metade da área de sobreposição
    # Se número for fracionário, a metade esquerda será diferente da metade direita
    half_overlap = overlap/2
    left_half = round(half_overlap)
    right_half = overlap - left_half
    up_half = left_half
    down_half = right_half
    
    # É o quanto será determinado, em pixels, pelo primeiro patch (patch da esquerda),
    # já que metade da parte de sobreposição
    # (a metade da direita) é desprezada
    # right_shift é o quanto é determinado pelo patch da direita
    # up_shift é o análogo da esquerda na vertical, portanto seria o patch de cima,
    # enquanto down_shift é o análogo da direita na vertical, portanto seria o patch de baixo
    left_shift = notoverlap + left_half
    right_shift = notoverlap + right_half
    up_shift = left_shift 
    down_shift = right_shift
    
    
    # Cria mosaico que será usado para escrever saída
    # Mosaico tem mesmas dimensões da referência usada para extração 
    # dos patches
    pred_test_mosaic = np.zeros(reference_shape)  

    # Dimensões máximas na vertical e horizontal
    ymax, xmax = pred_test_mosaic.shape
    
    # Linha e coluna de referência que serão atualizados no loop
    y = 0 
    x = 0 
    
    # Loop para escrever o mosaico    
    for patch in reference_batches:
        # Parte do mosaico a ser atualizada (debug)
        mosaico_parte_atualizada = pred_test_mosaic[y : y + p_size, x : x + p_size]
        print(mosaico_parte_atualizada)
        
        # Se reconstituímos os patches de borda (border_patches=True)
        # e se o patch é um patch que transborda a imagem, então vamos usar apenas
        # o espaço necessário no patch, com demarcação final por yp_limit e xp_limit
        if border_patches:
            xp_limit, yp_limit = calculate_xp_yp_limits(p_size, x, y, xmax, ymax, left_half, up_half, right_shift, down_shift)
        
        else:
            # Patches sempre são considerados por inteiro, pois não há patches de borda (border_patches=False) 
            yp_limit = p_size
            xp_limit = p_size
            
                    
                    
        # Se é primeiro patch, então vai usar todo patch
        # Do segundo patch em diante, vai ser usado a parte direita do patch, 
        # sobreescrevendo o patch anterior na área correspondente
        # Isso também acontece para os patches na vertical, sendo que o de baixo sobreescreverá o de cima
        if y == 0:
            if x == 0:
                pred_test_mosaic[y : y + p_size, x : x + p_size] = patch[0 : yp_limit, 0 : xp_limit]
            else:
                pred_test_mosaic[y : y + p_size, x : x + right_shift] = patch[0 : yp_limit, left_half : xp_limit]
        # y != 0
        else:
            if x == 0:
                pred_test_mosaic[y : y + down_shift, x : x + p_size] = patch[up_half : yp_limit, 0 : xp_limit]
            else:
                pred_test_mosaic[y : y + down_shift, x : x + right_shift] = patch[up_half : yp_limit, left_half : xp_limit]
            
            
        print(pred_test_mosaic) # debug
        
        
        # Incrementa linha de referência, de forma análoga ao incremento da coluna, se ela já foi esgotada 
        
        # No caso de não haver patches de bordas, é preciso testar se o próximo patch ultrapassará a borda 
        if not border_patches:
            if x == 0 and x + left_shift + right_shift > xmax:
                x = 0
        
                if y == 0:
                    y = y + up_shift
                else:
                    y = y + notoverlap
        
                continue
            
            else:
                if x + right_shift + right_shift > xmax:
                    x = 0
            
                    if y == 0:
                        y = y + up_shift
                    else:
                        y = y + notoverlap
            
                    continue
                
        
        
        if x == 0 and x + left_shift >= xmax:
            x = 0
            
            if y == 0:
                y = y + up_shift
            else:
                y = y + notoverlap
            
            continue
            
        elif x + right_shift >= xmax:
            x = 0
            
            if y == 0:
                y = y + up_shift
            else:
                y = y + notoverlap
            
            continue
        
        # Incrementa coluna de referência
        # Se ela for o primeiro patch (o mais a esquerda), então será considerada como patch da esquerda
        # Do segundo patch em diante, eles serão considerados como patch da direita
        if x == 0:
            x = x + left_shift
        else:
            x = x + notoverlap
            
    
    return pred_test_mosaic




# Faz um buffer em um array binário
def array_buffer(array, dist_cells=3):
    # Inicializa matriz resultado
    out_array = np.zeros_like(array)
    
    # Dimensões da imagem
    row = array.shape[0]
    col = array.shape[1]    
    
    # i designará o índice correspondete à distância horizontal (olunas) percorrida no buffer para cada célula  
    # j designará o índice correspondete à distância vertical (linhas) percorrida no buffer para cada célula  
    # h percorre colunas
    # k percorre linhas
    i, j, h, k = 0, 0, 0, 0
    
    # Array bidimensional
    if len(out_array.shape) < 3: 
        # Percorre matriz coluna a coluna
        # Percorre colunas
        while(h < col):
            k = 0
            # Percorre linhas
            while (k < row):
                # Teste se célula é maior ou igual a 1
                if (array[k][h] == 1):
                    i = h - dist_cells
                    while(i <= h + dist_cells and i < col):
                        if i < 0:
                            i+=1
                            continue
                            
                        j = k - dist_cells
                        while(j <= k + dist_cells and j < row):
                            if j < 0:
                                j+=1
                                continue
                            
                            # Testa se distância euclidiana do pixel está dentro do buffer
                            if ((i - h)**2 + (j - k)**2 <= dist_cells**2):
                                # Atualiza célula 
                                out_array[j][i] = array[k][h]
    
                            j+=1
                        i+=1
                k+=1
            h+=1
           
    # Array tridimensional disfarçado (1 na última dimensão)
    else:
        # Percorre matriz coluna a coluna
        # Percorre colunas
        while(h < col):
            k = 0
            # Percorre linhas
            while (k < row):
                # Teste se célula é maior ou igual a 1
                if (array[k][h][0] == 1):
                    i = h - dist_cells
                    while(i <= h + dist_cells and i < col):
                        if i < 0:
                            i+=1
                            continue
                            
                        j = k - dist_cells
                        while(j <= k + dist_cells and j < row):
                            if j < 0:
                                j+=1
                                continue
                            
                            # Testa se distância euclidiana do pixel está dentro do buffer
                            if ((i - h)**2 + (j - k)**2 <= dist_cells**2):
                                # Atualiza célula 
                                out_array[j][i][0] = array[k][h][0]
    
                            j+=1
                        i+=1
                k+=1
            h+=1
    
    # Retorna resultado
    return out_array


# Faz buffer em um raster .tif e salva o resultado como outro arquivo .tif
def buffer_binary_raster(in_raster_path, out_raster_path, dist_cells=3):
    # Lê raster como array numpy
    ds_raster = gdal.Open(in_raster_path)
    array = ds_raster.ReadAsArray()
    
    # Faz buffer de 3 pixels
    buffer_array_3px = array_buffer(array)
    
    # Exporta buffer para visualização
    driver = gdal.GetDriverByName('GTiff') # Geotiff Driver

    xsize = ds_raster.RasterXSize
    ysize = ds_raster.RasterYSize
    buffer_file = driver.Create(out_raster_path, xsize, ysize, bands=1, eType=gdal.GDT_Byte)

    buffer_band = buffer_file.GetRasterBand(1) 
    buffer_band.WriteArray(buffer_array_3px)

    buffer_file.SetGeoTransform(ds_raster.GetGeoTransform())
    buffer_file.SetProjection(ds_raster.GetProjection())    
    
    buffer_file.FlushCache()


# Faz buffer nos patches, sendo que o buffer é feito percorrendo os patches e fazendo o buffer,
# um por um
def buffer_patches(patch_test, dist_cells=3):
    result = []
    
    for i in range(len(patch_test)):
        print('Buffering patch {}/{}'.format(i+1, len(patch_test))) 
        # Patch being buffered 
        patch_batch = patch_test[i, ..., 0]
        
        # Do buffer
        patch_batch_r_new = array_buffer(patch_batch, dist_cells=dist_cells)[np.newaxis, ..., np.newaxis]

        # Append to result list       
        result.append(patch_batch_r_new)
        
    # Concatenate result patches to form result
    result = np.concatenate(result, axis=0)
            
    return result


# Função para salvar mosaico como tiff georreferenciado a partir de outra imagem do qual extraímos as informações geoespaciais
def save_raster_reference(in_raster_path, out_raster_path, array_exported, is_float=False):
    # Copia dados da imagem de labels
    ds_raster = gdal.Open(in_raster_path)
    xsize = ds_raster.RasterXSize
    ysize = ds_raster.RasterYSize
    
    # Exporta imagem para visualização
    driver = gdal.GetDriverByName('GTiff') # Geotiff Driver
    if is_float:
        file = driver.Create(out_raster_path, xsize, ysize, bands=1, eType=gdal.GDT_Float32) # Tipo float para números entre 0 e 1
    else:        
        file = driver.Create(out_raster_path, xsize, ysize, bands=1, eType=gdal.GDT_Byte) # Tipo Byte para somente números 0 e 1

    file_band = file.GetRasterBand(1) 
    file_band.WriteArray(array_exported)

    file.SetGeoTransform(ds_raster.GetGeoTransform())
    file.SetProjection(ds_raster.GetProjection())    
    
    file.FlushCache()
    
    
# Faz a previsão de todos os patches de treinamento
def Test(model, patch_test):
    result = model.predict(patch_test)
    predicted_classes = np.argmax(result, axis=-1)
    return predicted_classes

# Faz a previsão de todos os patches de treinamento aos poucos (em determinados lotes).
# Isso é para poupar o processamento do computador
# Retorna também a probabilidade das classes, além da predição
def Test_Step(model, patch_test, step=2, out_sigmoid=False, threshold_sigmoid=0.5):
    result = model.predict(patch_test, batch_size=step, verbose=1)

    if out_sigmoid:
        predicted_classes = np.where(result > threshold_sigmoid, 1, 0)
    else:
        predicted_classes = np.argmax(result, axis=-1)[..., np.newaxis]
        
    return predicted_classes, result


# Faz a previsão de todos os patches de treinamento aos poucos (em determinados lotes).
# Isso é para poupar o processamento do computador
# Retorna também a probabilidade das classes, além da predição
# Retorna uma previsões para todo o ensemble de modelos, dado pela lista de modelos passada
# como parâmetro
def Test_Step_Ensemble(model_list, patch_test, step, out_sigmoid=False, threshold_sigmoid=0.5):
    # Lista de predições, cada item é a predição para um modelo do ensemble  
    pred_list = []
    prob_list = []
        
    # Loop no modelo 
    for i, model in enumerate(model_list):
        print(f'Predizendo usando modelo número {i} \n\n')
        predicted_classes, result = Test_Step(model, patch_test, step, out_sigmoid=out_sigmoid, threshold_sigmoid=threshold_sigmoid)
        pred_list.append(predicted_classes)
        prob_list.append(result)
        
    # Transforma listas em arrays e retorna arrays como resultado
    pred_array = np.array(pred_list)
    prob_array = np.array(prob_list)
    
    return pred_array, prob_array

# Função que calcula algumas métricas: acurácia, f1-score, sensibilidade e precisão 
def compute_metrics(true_labels, predicted_labels):
    accuracy = 100*accuracy_score(true_labels, predicted_labels)
    f1score = 100*f1_score(true_labels, predicted_labels, average=None)
    recall = 100*recall_score(true_labels, predicted_labels, average=None)
    precision = 100*precision_score(true_labels, predicted_labels, average=None)
    return accuracy, f1score, recall, precision




        
# Função que adiciona o Canal das probabilidades de 0 (fundo) para a Referência.
# Isso porque a Referência vem com apenas um canal, que é o de Estradas, onde Estradas é 1 e o
# restante é 0.
# Isso se faz necessário para compilar o modelo com a métrica Recall
def adiciona_prob_0(y_prob_1):
    # Adiciona um Canal ao longo da última dimensão, ou -1, com Zeros para depois editar
    # Esse canal vai ficar no canal 0, portanto antes do canal do array existente
    # Por isso a forma de concatenação é concatenar o canal de 0s com o array existente 
    zeros = np.zeros(y_prob_1.shape, dtype=np.uint8)
    y_prob_1 = np.concatenate((zeros, y_prob_1), axis=-1)
    
    # Percorre dimensão dos batches e adiciona imagens (arrays) que são o complemento
    # da probabilidade de estradas
    for i in range(y_prob_1.shape[0]):
        y_prob_1[i, :, :, 0] = 1 - y_prob_1[i, :, :, 1]
        
    return y_prob_1



m_p = tf.keras.metrics.Precision()
m_r = tf.keras.metrics.Recall()

def my_f1score(y_true, y_pred): 
    # Get only the result of class 1 (roads)
    y_pred = y_pred[..., 1:2]    
    
    # Flatten arrays
    y_true_flat = tf.reshape(y_true, (tf.shape(y_true)[0] * tf.shape(y_true)[1] * tf.shape(y_true)[2], 1))
    y_pred_flat = tf.reshape(y_pred, (tf.shape(y_pred)[0] * tf.shape(y_pred)[1] * tf.shape(y_pred)[2], 1))
    
    # Calculate precision and recall
    m_p.update_state(y_true_flat, y_pred_flat)
    m_r.update_state(y_true_flat, y_pred_flat)
    
    precision = float(m_p.result())
    recall = float(m_r.result())

    # Calculate f1-score
    f1score = 2  *  (precision*recall) / (precision+recall)
    
    return f1score


class F1Score(Metric):
    def __init__(self, name='f1score', beta=1, threshold=0.5, epsilon=1e-7, **kwargs):
        # initializing an object of the super class
        super(F1Score, self).__init__(name=name, **kwargs)
          
        # initializing state variables
        self.tp = self.add_weight(name='tp', initializer='zeros') # initializing true positives 
        self.actual_positive = self.add_weight(name='fp', initializer='zeros') # initializing actual positives
        self.predicted_positive = self.add_weight(name='fn', initializer='zeros') # initializing predicted positives
          
        # initializing other atrributes that wouldn't be changed for every object of this class
        self.beta_squared = beta**2 
        self.threshold = threshold
        self.epsilon = epsilon
    
    def update_state(self, ytrue, ypred, sample_weight=None):
        # Pega só referente a classe 1
        ypred = ypred[..., 1:2] 
        ytrue = ytrue[..., 1:2] 
          
        # casting ytrue and ypred as float dtype
        ytrue = tf.cast(ytrue, tf.float32)
        ypred = tf.cast(ypred, tf.float32)
        
        #print(f'Shape Shape de Y True é {ytrue.shape}')
        #print(f'Shape Shape de Y Pred é {ytrue.shape}')
          
        # setting values of ypred greater than the set threshold to 1 while those lesser to 0
        ypred = tf.cast(tf.greater_equal(ypred, tf.constant(self.threshold)), tf.float32)
            
        self.tp.assign_add(tf.reduce_sum(ytrue*ypred)) # updating true positives atrribute
        self.predicted_positive.assign_add(tf.reduce_sum(ypred)) # updating predicted positive atrribute
        self.actual_positive.assign_add(tf.reduce_sum(ytrue)) # updating actual positive atrribute
    
    def result(self):
        self.precision = self.tp/(self.predicted_positive+self.epsilon) # calculates precision
        self.recall = self.tp/(self.actual_positive+self.epsilon) # calculates recall
          
        # calculating fbeta
        self.fb = (1+self.beta_squared)*self.precision*self.recall / (self.beta_squared*self.precision + self.recall + self.epsilon)
        
        return self.fb
    
    def reset_state(self):
        self.tp.assign(0) # resets true positives to zero
        self.predicted_positive.assign(0) # resets predicted positives to zero
        self.actual_positive.assign(0) # resets actual positives to zero
        




def dice_loss(y_true, y_pred, smooth=1e-6):
    '''
    Dice Coef = (2*Inter)/(Union+Inter)
    Dice Loss = 1 - Dice Coef    
    '''
    # Cast y_true as float 32
    y_true = tf.cast(y_true, tf.float32)
    
    # Flatten Arrays        
    y_true_f = K.flatten(y_true[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])
    
    # Intersection and Union
    intersection = K.sum(y_true_f * y_pred_f)
    union_plus_inter = K.sum(y_true_f) + K.sum(y_pred_f)
    
    # Compute Dice Loss
    dice_coef = (2. * intersection + smooth) / (union_plus_inter + smooth)
    #dice_loss = 1 - dice_coef
    dice_loss = -K.log(dice_coef)
    
    return dice_loss

        
def weighted_focal_loss(alpha, gamma=0):
    """
    compute focal loss according to the prob of the sample.
    loss= -(1-p)^gamma*log(p)

    Variables:
        gamma: integer
        alpha: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        gamma = 2 #The larger the gamma value, the less importance of easily classified. Typical values: 0 to 5
        alpha = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_focal_loss(gamma, alpha)
        model.compile(loss=focal_loss,optimizer='adam')
    """
    
    alpha = K.variable(alpha)
        
    def focal_loss(y_true, y_pred):
        print('Tipo y_true =', y_true.dtype)
        print('Tipo y_pred =', y_pred.dtype)
        print('Tipo alpha =', alpha.dtype)
        # Cast y_true as float 32
        y_true = tf.cast(y_true, tf.float32)
        
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        y_pred_inv = 1.0 - y_pred
        y_pred_inv = K.pow(y_pred_inv, gamma)
        # calc
        focal_loss = y_true * K.log(y_pred) * y_pred_inv * alpha
        focal_loss = K.mean(-K.sum(focal_loss, -1))
        return focal_loss
    
    return focal_loss


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    #weights = K.variable(weights)
    weights = tf.Variable(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        #y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        #y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        #loss = y_true * K.log(y_pred) * weights
        loss = y_true * tf.math.log(y_pred) * weights
        #loss = K.mean(-K.sum(loss, -1))
        loss = tf.reduce_mean(-tf.reduce_sum(loss, axis=-1))
        
        return loss

    return loss


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


# Função para treinar o modelo conforme os dados (arrays numpy) em uma pasta de entrada, salvando o modelo e o 
# histórico na pasta de saída
def treina_modelo(input_dir: str, y_dir: str, output_dir: str, model_type: str ='resunet chamorro', epochs=150, early_stopping=True, 
                  early_loss=False, loss='cross', weights=[0.25, 0.75], gamma=2, metric=F1Score(), best_model_filename = 'best_model',
                  train_with_dataset=False, lr_decay=False):
    '''
    

    e talvez correction=False
    Parameters
    ----------
    input_dir : str
        Relative Path for Input Folder, e.g, r"entrada/".
    output_dir : str
        Relative Path for Output Folder, e.g, r"resultados/".
    model_type : str, optional
        DESCRIPTION. The default is 'resunet'.
     : TYPE
        DESCRIPTION.
    lr_decay=Dicionário, por exemplo, {initial_lrate = 0.1,
             drop = 0.5,
             epochs_drop = 10.0}, ou None, para sem learning rate decay 

    Returns
    -------
    None.

    '''
    # TODO
    # Completar código para train_with_dataset
    
    # Marca tempo total do treinamento. Começa Contagem
    start = time.time()
    
    # Lê arrays salvos
    '''
    if correction:
        x_train = np.load(input_dir + 'pred_train.npy')
        x_valid = np.load(input_dir + 'pred_valid.npy')
    else:
        x_train = np.load(input_dir + 'x_train.npy')
        x_valid = np.load(input_dir + 'x_valid.npy')
    '''
    if train_with_dataset:
        # Load Datasets
        train_dataset = tf.data.Dataset.load(input_dir + 'train_dataset/')
        valid_dataset = tf.data.Dataset.load(input_dir + 'valid_dataset/')
        test_dataset = tf.data.Dataset.load(input_dir + 'test_dataset/')
        
        print('Shape dos arrays:')
        print('Shape X Train Dataset: ', (train_dataset.cardinality().numpy(),)+
              iter(train_dataset).next()[0].numpy().shape)
        print('Shape Y Train Dataset: ', (train_dataset.cardinality().numpy(),)+
              iter(train_dataset).next()[1].numpy().shape)
        print('Shape X Valid Dataset: ', (train_dataset.cardinality().numpy(),)+
              iter(valid_dataset).next()[0].numpy().shape)
        print('Shape Y Valid Dataset: ', (train_dataset.cardinality().numpy(),)+
              iter(valid_dataset).next()[1].numpy().shape)
        print('')
        
    else:
        x_train = np.load(input_dir + 'x_train.npy')
        x_valid = np.load(input_dir + 'x_valid.npy')
        y_train = np.load(y_dir + 'y_train.npy')
        y_valid = np.load(y_dir + 'y_valid.npy')
        
        # Faz codificação One-Hot dos Ys
        y_train = to_categorical(y_train, num_classes=2, dtype='uint8')
        y_valid = to_categorical(y_valid, num_classes=2, dtype='uint8')
        
        print('Shape dos arrays:')
        print('Shape x_train: ', x_train.shape)
        print('Shape y_train: ', y_train.shape)
        print('Shape x_valid: ', x_valid.shape)
        print('Shape y_valid: ', y_valid.shape)
        print('')

        # Transforma dados para tensores dentro da CPU para evitar falta de espaço na GPU
        if not train_with_dataset:
            with tf.device('/CPU:0'):
                x_train = tf.convert_to_tensor(x_train)
                y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
                x_valid = tf.convert_to_tensor(x_valid)
                y_valid = tf.convert_to_tensor(y_valid, dtype=tf.float32)
        
        # Livra memória no que for possível
        gc.collect()  

    
    # Constroi modelo
    if train_with_dataset:
        # input_shape = (patch_size, patch_size, image_channels)
        input_shape = iter(train_dataset).next()[0].numpy().shape
        num_classes = 2
        model = build_model(input_shape, num_classes, model_type=model_type)
        # model.summary()
        print('Input Patch Shape = ', input_shape)
        print()
        
    else:
        # input_shape = (patch_size, patch_size, image_channels)
        input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
        num_classes = 2
        model = build_model(input_shape, num_classes, model_type=model_type)
        # model.summary()
        print('Input Patch Shape = ', input_shape)
        print()
        
    # Compila o modelo
    learning_rate = 0.001 # Learning Rate
    #learning_rate = 0.01 # Learning Rate # não estava convergindo na primeira rede
    optimizer = Adam(learning_rate = learning_rate , beta_1=0.9) # Otimizador
    #optimizer = SGD(learning_rate = learning_rate , momentum=0.9) # Otimizador

    if loss == 'focal':
        focal_loss = CategoricalFocalCrossentropy(alpha=weights, gamma=gamma)
        #focal_loss = weighted_focal_loss(alpha=weights, gamma=gamma)
        model.compile(loss = focal_loss, optimizer=optimizer , metrics=[metric, tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)])
    elif loss == 'cross':
        #model.compile(loss = 'sparse_categorical_crossentropy', optimizer=adam, metrics=[metric])
        model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics=[metric, tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)])
    elif loss == 'mse':
        optimizer = SGD(learning_rate = learning_rate)
        model.compile(loss = 'mse', optimizer=optimizer, metrics=[metric, tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)])
    elif loss == 'dice':
        model.compile(loss = dice_loss, optimizer=optimizer, metrics=[metric, tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)])
    elif loss == 'jaccard':
        model.compile(loss = jaccard_distance_loss, optimizer=optimizer, metrics=[metric, tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)])
    elif loss == 'wcross':
        model.compile(loss = weighted_categorical_crossentropy(weights), optimizer=optimizer, metrics=[metric, tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1)])
        
        
     # Definição dos Outros hiperparâmetros
    batch_size = 16
    
    epochs = epochs

    # Parâmetros do Early Stopping
    early_stopping = early_stopping
    early_stopping_epochs = 50
    #early_stopping_epochs = 2
    #early_stopping_delta = 0.001 # aumento delta (percentual de diminuição da perda) equivalente a 0.1%
    early_stopping_delta = 0.001 # aumento delta (percentual de diminuição da perda) equivalente a 0.1%

    if train_with_dataset:
        shuffle_buffer = 100
        train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer).batch(batch_size=batch_size).prefecth(buffer_size=1)
    
    print('Hiperparâmetros:')
    print('Modelo:', model_type)
    print('Batch Size:', batch_size)
    print('Epochs:', epochs)
    print('Early Stopping:', early_stopping)
    print('Early Stopping Epochs:', early_stopping_epochs)
    print('Early Stopping Delta:', early_stopping_delta)
    print('Otimizador:', 'Adam')
    print('Learning Rate:', learning_rate)
    print('Beta 1:', 0.9)
    print('Função de Perda:', loss)
    print('Gamma para Focal Loss:', gamma)
    print()
        
    # Nome do modelo a ser salvo
    best_model_filename = best_model_filename
    

    
    # Treina o modelo
    # Testa se a métrica é string
    if isinstance(metric, str):
        history = train_unet(model, x_train, y_train, x_valid, y_valid, batch_size, epochs, early_stopping_epochs, early_stopping_delta, 
                     output_dir, best_model_filename, early_stopping=early_stopping, early_loss=early_loss, 
                     metric_name=metric, lr_decay=lr_decay)
    
    # Testa se é instância
    elif isinstance(metric, object):
        history = train_unet(model, x_train, y_train, x_valid, y_valid, batch_size, epochs, early_stopping_epochs, early_stopping_delta, 
                         output_dir, best_model_filename, early_stopping=early_stopping, early_loss=early_loss, 
                         metric_name=metric.__class__.__name__.lower(), lr_decay=lr_decay)
        
    # Testa se é função
    elif isinstance(metric, (types.FunctionType, types.BuiltinFunctionType)):
        history = train_unet(model, x_train, y_train, x_valid, y_valid, batch_size, epochs, early_stopping_epochs, early_stopping_delta, 
                         output_dir, best_model_filename, early_stopping=early_stopping, early_loss=early_loss, 
                         metric_name=metric.__name__, lr_decay=lr_decay)

        

    # Imprime e salva história do treinamento
    print('history = \n', history)
    
    # Marca tempo total do treinamento. Encerra Contagem
    end = time.time()
    
    # Salva histórico e inclui no arquivo texto a contagem de tempo gasto
    with open(os.path.join(output_dir, 'history_' + best_model_filename + '.txt'), 'w') as f:
        f.write('history = \n')
        f.write(str(history))
        f.write(f'\nTempo total gasto no treinamento foi de {end-start} segundos')
        
    with open(os.path.join(output_dir, 'history_pickle_' +  best_model_filename + '.pickle'), "wb") as fp: # Salva histórico (lista Python) para recuperar depois
        pickle.dump(history, fp)
        
    # Mostra histórico em gráfico
    show_graph_loss_accuracy(np.asarray(history), 1, metric_name='F1-Score', save=True, save_path=output_dir)
    
    # Escreve hiperparâmetreos e modelo usados no Diretório
    with open(os.path.join(output_dir, 'model_configuration_used.txt'), 'w') as f:
        f.writelines([f'Model Type = {model_type}\n', f'Batch Size = {batch_size}\n', f'Epochs = {epochs}\n',
                      f'Early Stopping Epochs = {early_stopping_epochs}\n', f'Early Stopping Delta = {early_stopping_delta}\n',
                      f'Learning Rate = {learning_rate}\n', f'Função de perda = {loss}\n', f'Optimizer = {str(type(optimizer))}\n'])
        if model_type == 'unet transformer':
            f.write(str(config_dict) + '\n')
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    
    
        




def treina_modelo_ensemble(input_dir: str, output_dir_ensemble: str, n_members: int = 10, model_type: str ='resunet', epochs=150, 
                           early_stopping=True, early_loss=True, loss='focal', gamma=2, metric=F1Score(), 
                           best_model_filename = 'best_model'):
    # Loop para treinar vários modelos diversas vezes
    for i in range(n_members):
        treina_modelo(input_dir, output_dir_ensemble, model_type=model_type, epochs=epochs, early_stopping=early_stopping, 
                      early_loss=early_loss, loss=loss, gamma=gamma, metric=metric, 
                      best_model_filename=best_model_filename + '_' + str(i+1))


        

def plota_resultado_experimentos(lista_precisao, lista_recall, lista_f1score, lista_nomes_exp, nome_conjunto,
                                 save=False, save_path=r''):
    '''
    

    Parameters
    ----------
    lista_precisao : TYPE
        Lista das Precisões na ordem dada pela lista dos experimentos lista_nomes_exp.
    lista_recall : TYPE
        Lista dos Recalls na ordem dada pela lista dos experimentos lista_nomes_exp.
    lista_f1score : TYPE
        Lista dos F1-Scores na ordem dada pela lista dos experimentos lista_nomes_exp.
    lista_nomes_exp : TYPE
        Lista dos nomes dos experimentos.
    nome_conjunto : TYPE
        Nome do Conjunto sendo testado, por exemplo, "Treino", "Validação", "Teste", "Mosaicos de Teste", etc. 
        Esse nome será usado no título do gráfico e para nome do arquivo do gráfico.

    Returns
    -------
    None.

    '''
    # Figura, posição dos labels e largura da barra
    n_exp = len(lista_nomes_exp)
    fig = plt.figure(figsize = (15, 10))
    x = np.arange(n_exp) # the label locations
    width = 0.2
    
    # Cria barras
    plt.bar(x-0.2, lista_precisao, width, color='cyan')
    plt.bar(x, lista_recall, width, color='orange')
    plt.bar(x+0.2, lista_f1score, width, color='green')
    
    # Nomes dos grupos em X
    plt.xticks(x, lista_nomes_exp)
    
    # Nomes dos eixos
    plt.xlabel("Métricas")
    plt.ylabel("Valor Percentual (%)")
    
    # Título do gráfico
    plt.title("Resultados das Métricas para %s" % (nome_conjunto))
    
    # Legenda
    legend = plt.legend(["Precisão", "Recall", "F1-Score"], ncol=3, framealpha=0.5)
    #legend.get_frame().set_alpha(0.9)

    # Salva figura se especificado
    if save:
        plt.savefig(save_path + nome_conjunto + ' plotagem_resultado.png')
    
    # Exibe gráfico
    plt.show()
    

# Gera gráficos de Treino, Validação, Teste e Mosaicos de Teste para os experimentos desejados,
# devendo informar os diretórios com os resultados desses experimentos, o nome desses experimentos e
# o diretório onde salvar os gráficos
def gera_graficos(metrics_dirs_list, lista_nomes_exp, save_path=r''):
    # Open metrics and add to list
    resultados_metricas_list = []
    
    for mdir in metrics_dirs_list:
        with open(mdir + "resultados_metricas.pickle", "rb") as fp:   # Unpickling
            metrics = pickle.load(fp)
            
        resultados_metricas_list.append(metrics)
        
    # Resultados para Treino
    precision_treino = [] 
    recall_treino = []
    f1score_treino = []
    
    # Resultados para Validação
    precision_validacao = []
    recall_validacao = []
    f1score_validacao = []
    
    # Resultados para Teste
    precision_teste = []
    recall_teste = []
    f1score_teste = []
    
    # Resultados para Mosaicos de Teste
    precision_mosaicos_teste = []
    recall_mosaicos_teste = [] 
    f1score_mosaicos_teste = []
    
    # Loop inside list of metrics and add to respective list
    for resultado in resultados_metricas_list:
        # Append results to Train
        precision_treino.append(resultado['relaxed_precision_train'])
        recall_treino.append(resultado['relaxed_recall_train'])
        f1score_treino.append(resultado['relaxed_f1score_train'])
        
        # Append results to Valid
        precision_validacao.append(resultado['relaxed_precision_valid'])
        recall_validacao.append(resultado['relaxed_recall_valid'])
        f1score_validacao.append(resultado['relaxed_f1score_valid'])
        
        # Append results to Test
        precision_teste.append(resultado['relaxed_precision_test'])
        recall_teste.append(resultado['relaxed_recall_test'])
        f1score_teste.append(resultado['relaxed_f1score_test'])
        
        # Append results to Mosaics of Test
        precision_mosaicos_teste.append(resultado['relaxed_precision_mosaics'])
        recall_mosaicos_teste.append(resultado['relaxed_recall_mosaics'])
        f1score_mosaicos_teste.append(resultado['relaxed_f1score_mosaics'])
    
    
    # Gera gráficos
    plota_resultado_experimentos(precision_treino, recall_treino, f1score_treino, lista_nomes_exp, 
                             'Treino', save=True, save_path=save_path)
    plota_resultado_experimentos(precision_validacao, recall_validacao, f1score_validacao, lista_nomes_exp, 
                             'Validação', save=True, save_path=save_path)
    plota_resultado_experimentos(precision_teste, recall_teste, f1score_teste, lista_nomes_exp, 
                                 'Teste', save=True, save_path=save_path)
    plota_resultado_experimentos(precision_mosaicos_teste, recall_mosaicos_teste, f1score_mosaicos_teste, lista_nomes_exp, 
                                 'Mosaicos de Teste', save=True, save_path=save_path)


# Avalia um modelo segundo conjuntos de treino, validação, teste e mosaicos de teste
def avalia_modelo(input_dir: str, y_dir: str, output_dir: str, metric_name = 'F1-Score', 
                  dist_buffers=[3], avalia_train=False, avalia_diff=False, avalia_ate_teste=False):
    metric_name = metric_name
    dist_buffers = dist_buffers
    
    # Nome do modelo salvo
    best_model_filename = 'best_model'
    
    # Lê histórico
    with open(output_dir + 'history_pickle_' + best_model_filename + '.pickle', "rb") as fp:   
        history = pickle.load(fp)
    
    # Mostra histórico em gráfico
    show_graph_loss_accuracy(np.asarray(history), 1, metric_name = metric_name, save=True, save_path=output_dir)

    # Load model
    model = load_model(output_dir + best_model_filename + '.h5', compile=False, custom_objects={"Patches": Patches, 
                                                                                                "MixVisionTransformer": MixVisionTransformer,
                                                                                                "SegFormerHead": SegFormerHead,
                                                                                                "ResizeLayer": ResizeLayer})

    # Avalia treino    
    if avalia_train:
        x_train = np.load(input_dir + 'x_train.npy')
        y_train = np.load(y_dir + 'y_train.npy')

        if not os.path.exists(os.path.join(output_dir, 'pred_train.npy')) or \
           not os.path.exists(os.path.join(output_dir, 'prob_train.npy')): 
    
            pred_train, prob_train = Test_Step(model, x_train, 2)
            
            # Probabilidade apenas da classe 1 (de estradas)
            prob_train = prob_train[..., 1:2] 
            
            # Converte para tipos que ocupam menos espaço
            pred_train = pred_train.astype(np.uint8)
            prob_train = prob_train.astype(np.float16)

            # Salva arrays de predição do Treinamento e Validação    
            salva_arrays(output_dir, pred_train=pred_train, prob_train=prob_train)
        
        pred_train = np.load(output_dir + 'pred_train.npy')
        prob_train = np.load(output_dir + 'prob_train.npy')

        # Faz os Buffers necessários, para treino, nas imagens 
        
        # Precisão Relaxada - Buffer na imagem de rótulos
        # Sensibilidade Relaxada - Buffer na imagem extraída
        # F1-Score Relaxado - É obtido através da Precisão e Sensibilidade Relaxadas
        
        buffers_y_train = {}
        buffers_pred_train = {}
    
        for dist in dist_buffers:
            # Buffers para Precisão Relaxada
            if not os.path.exists(os.path.join(y_dir, f'buffer_y_train_{dist}px.npy')): 
                buffers_y_train[dist] = buffer_patches(y_train, dist_cells=dist)
                np.save(y_dir + f'buffer_y_train_{dist}px.npy', buffers_y_train[dist])
                
            # Buffers para Sensibilidade Relaxada
            if not os.path.exists(os.path.join(output_dir, f'buffer_pred_train_{dist}px.npy')):
                buffers_pred_train[dist] = buffer_patches(pred_train, dist_cells=dist)
                np.save(output_dir + f'buffer_pred_train_{dist}px.npy', buffers_pred_train[dist])
            

        # Lê buffers de arrays de predição do Treinamento
        for dist in dist_buffers:
            buffers_y_train[dist] = np.load(y_dir + f'buffer_y_train_{dist}px.npy')            
            buffers_pred_train[dist] = np.load(output_dir + f'buffer_pred_train_{dist}px.npy')
  
        
        # Relaxed Metrics for training
        relaxed_precision_train, relaxed_recall_train, relaxed_f1score_train = {}, {}, {}
    
    
        for dist in dist_buffers:
            with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
                relaxed_precision_train[dist], relaxed_recall_train[dist], relaxed_f1score_train[dist] = compute_relaxed_metrics(y_train, 
                                                                                                                                     pred_train, buffers_y_train[dist],
                                                                                                                                     buffers_pred_train[dist], 
                                                                                                                                     nome_conjunto = 'Treino', 
                                                                                                                                     print_file=f)        
           
    
        # Release Memory
        x_train = None
        y_train = None
        pred_train = None
        prob_train = None
        
        buffers_y_train = None
        buffers_pred_train = None
            
        gc.collect()
    
    # Avalia Valid
    x_valid = np.load(input_dir + 'x_valid.npy')
    y_valid = np.load(y_dir + 'y_valid.npy')
    
    if not os.path.exists(os.path.join(output_dir, 'pred_valid.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'prob_valid.npy')):
           
           pred_valid, prob_valid = Test_Step(model, x_valid, 2)
           
           # Probabilidade apenas da classe 1 (de estradas)
           prob_valid = prob_valid[..., 1:2]
           
           # Converte para tipos que ocupam menos espaço
           pred_valid = pred_valid.astype(np.uint8)
           prob_valid = prob_valid.astype(np.float16)
           
           # Salva arrays de predição do Treinamento e Validação    
           salva_arrays(output_dir, pred_valid=pred_valid, prob_valid=prob_valid)
           
    pred_valid = np.load(output_dir + 'pred_valid.npy')
    prob_valid = np.load(output_dir + 'prob_valid.npy')
    
    buffers_y_valid = {}
    buffers_pred_valid = {}
    
    for dist in dist_buffers:
        # Buffers para Precisão Relaxada
        if not os.path.exists(os.path.join(y_dir, f'buffer_y_valid_{dist}px.npy')): 
            buffers_y_valid[dist] = buffer_patches(y_valid, dist_cells=dist)
            np.save(y_dir + f'buffer_y_valid_{dist}px.npy', buffers_y_valid[dist])
        
        # Buffers para Sensibilidade Relaxada
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_valid_{dist}px.npy')): 
            buffers_pred_valid[dist] = buffer_patches(pred_valid, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_valid_{dist}px.npy', buffers_pred_valid[dist])
            
            
    for dist in dist_buffers:
        buffers_y_valid[dist] = np.load(y_dir + f'buffer_y_valid_{dist}px.npy')
        buffers_pred_valid[dist] = np.load(output_dir + f'buffer_pred_valid_{dist}px.npy')
        
        
    relaxed_precision_valid, relaxed_recall_valid, relaxed_f1score_valid = {}, {}, {}
    
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
            relaxed_precision_valid[dist], relaxed_recall_valid[dist], relaxed_f1score_valid[dist] = compute_relaxed_metrics(y_valid, 
                                                                                                           pred_valid, buffers_y_valid[dist],
                                                                                                           buffers_pred_valid[dist],
                                                                                                           nome_conjunto = 'Validação',
                                                                                                           print_file=f) 
            

    x_valid = None
    y_valid = None
    pred_valid = None
    prob_valid = None   
        
    buffers_y_valid = None
    buffers_pred_valid = None
        
    gc.collect()


    # Avalia teste
    x_test = np.load(input_dir + 'x_test.npy')
    y_test = np.load(y_dir + 'y_test.npy')
    
    if not os.path.exists(os.path.join(output_dir, 'pred_test.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'prob_test.npy')):
        pred_test, prob_test = Test_Step(model, x_test, 2)
        
        prob_test = prob_test[..., 1:2] # Essa é a probabilidade de prever estrada (valor de pixel 1)
    
        # Converte para tipos que ocupam menos espaço
        pred_test = pred_test.astype(np.uint8)
        prob_test = prob_test.astype(np.float16)
        
        # Salva arrays de predição do Teste. Arquivos da Predição (pred) são salvos na pasta de arquivos de saída (resultados_dir)
        salva_arrays(output_dir, pred_test=pred_test, prob_test=prob_test)
        
    pred_test = np.load(output_dir + 'pred_test.npy')
    prob_test = np.load(output_dir + 'prob_test.npy')
    
    buffers_y_test = {}
    buffers_pred_test = {}
    
    for dist in dist_buffers:
        # Buffer para Precisão Relaxada
        if not os.path.exists(os.path.join(y_dir, f'buffer_y_test_{dist}px.npy')):
            buffers_y_test[dist] = buffer_patches(y_test, dist_cells=dist)
            np.save(y_dir + f'buffer_y_test_{dist}px.npy', buffers_y_test[dist])
            
        # Buffer para Sensibilidade Relaxada
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_test_{dist}px.npy')):
            buffers_pred_test[dist] = buffer_patches(pred_test, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_test_{dist}px.npy', buffers_pred_test[dist])
            
            
    for dist in dist_buffers:
        buffers_y_test[dist] = np.load(y_dir + f'buffer_y_test_{dist}px.npy')
        buffers_pred_test[dist] = np.load(output_dir + f'buffer_pred_test_{dist}px.npy')
        
        
    relaxed_precision_test, relaxed_recall_test, relaxed_f1score_test = {}, {}, {}
    
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
            relaxed_precision_test[dist], relaxed_recall_test[dist], relaxed_f1score_test[dist] = compute_relaxed_metrics(y_test, 
                                                                                                       pred_test, buffers_y_test[dist],
                                                                                                       buffers_pred_test[dist], 
                                                                                                       nome_conjunto = 'Teste', 
                                                                                                       print_file=f)
            
    
    x_test = None
    y_test = None
    prob_test = None   
        
    buffers_y_test = None
    buffers_pred_test = None
        
    gc.collect()
    
    # Se avalia_ate_teste = True, a função para por aqui e não gera mosaicos e nem avalia a diferença
    # Os resultados no dicionário são atualizados até o que se tem, outras informações tem o valor None
    if avalia_ate_teste:
        if not avalia_train:
            relaxed_precision_train = None
            relaxed_recall_train = None
            relaxed_f1score_train = None

        relaxed_precision_mosaics = None
        relaxed_recall_mosaics = None
        relaxed_f1score_mosaics = None
        
        relaxed_precision_diff = None
        relaxed_recall_diff = None
        relaxed_f1score_diff = None
            
        dict_results = {
            'relaxed_precision_train': relaxed_precision_train, 'relaxed_recall_train': relaxed_recall_train, 'relaxed_f1score_train': relaxed_f1score_train,
            'relaxed_precision_valid': relaxed_precision_valid, 'relaxed_recall_valid': relaxed_recall_valid, 'relaxed_f1score_valid': relaxed_f1score_valid,
            'relaxed_precision_test': relaxed_precision_test, 'relaxed_recall_test': relaxed_recall_test, 'relaxed_f1score_test': relaxed_f1score_test,
            'relaxed_precision_mosaics': relaxed_precision_mosaics, 'relaxed_recall_mosaics': relaxed_recall_mosaics, 'relaxed_f1score_mosaics': relaxed_f1score_mosaics,
            'relaxed_precision_diff': relaxed_precision_diff, 'relaxed_recall_diff': relaxed_recall_diff, 'relaxed_f1score_diff': relaxed_f1score_diff
            }
        
        with open(output_dir + 'resultados_metricas' + '.pickle', "wb") as fp: # Salva dicionário com métricas do resultado do experimento
            pickle.dump(dict_results, fp)
        
        return dict_results
    
    # Gera Mosaicos de Teste
    # Avalia Mosaicos de Teste
    if not os.path.exists(os.path.join(y_dir, 'y_mosaics.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'pred_mosaics.npy')):

        # Stride e Dimensões do Tile
        with open(y_dir + 'info_tiles_test.pickle', "rb") as fp:   
            info_tiles_test = pickle.load(fp)
            len_tiles_test = info_tiles_test['len_tiles_test']
            shape_tiles_test = info_tiles_test['shape_tiles_test']
            patch_test_stride = info_tiles_test['patch_stride_test']
    
        # patch_test_stride = 210 # tem que estabelecer de forma manual de acordo com o stride usado para extrair os patches

        # labels_test_shape = (1500, 1500) # também tem que estabelecer de forma manual de acordo com as dimensões da imagem de referência
        labels_test_shape = shape_tiles_test

        # n_test_tiles = 49 # Número de tiles de teste
    
        # Pasta com os tiles de teste para pegar informações de georreferência
        #test_labels_tiles_dir = r'dataset_massachusetts_mnih/test/maps'
        test_labels_tiles_dir = r'tiles/masks/2018/test'
        labels_paths = [os.path.join(test_labels_tiles_dir, arq) for arq in os.listdir(test_labels_tiles_dir)]
        labels_paths.sort()
    
        # Gera mosaicos e lista com os mosaicos previstos
        pred_mosaics = gera_mosaicos(output_dir, pred_test, labels_paths, 
                                     patch_test_stride=patch_test_stride,
                                     labels_test_shape=labels_test_shape,
                                     len_tiles_test=len_tiles_test, is_float=False)
    
        # Lista e Array dos Mosaicos de Referência
        y_mosaics = [gdal.Open(y_mosaic_path).ReadAsArray() for y_mosaic_path in labels_paths]
        #y_mosaics = np.array(y_mosaics)[..., np.newaxis]
        y_mosaics = stack_uneven(y_mosaics)[..., np.newaxis]
        
        # Transforma valor NODATA de y_mosaics em 0
        y_mosaics[y_mosaics==255] = 0
        
        # Array dos Mosaicos de Predição 
        #pred_mosaics = np.array(pred_mosaics)[..., np.newaxis]
        pred_mosaics = stack_uneven(pred_mosaics)[..., np.newaxis]
        pred_mosaics = pred_mosaics.astype(np.uint8)
        
        # Salva Array dos Mosaicos de Predição
        salva_arrays(y_dir, y_mosaics=y_mosaics)
        salva_arrays(output_dir, pred_mosaics=pred_mosaics)


    # Libera memória se for possível
    pred_test = None
    gc.collect()
    
    # Lê Mosaicos 
    y_mosaics = np.load(y_dir + 'y_mosaics.npy')
    pred_mosaics = np.load(output_dir + 'pred_mosaics.npy')
        
    # Buffer dos Mosaicos de Referência e Predição
    buffers_y_mosaics = {}
    buffers_pred_mosaics = {}
    
    for dist in dist_buffers:
        # Buffer dos Mosaicos de Referência
        if not os.path.exists(os.path.join(y_dir, f'buffer_y_mosaics_{dist}px.npy')):
            buffers_y_mosaics[dist] = buffer_patches(y_mosaics, dist_cells=dist)
            np.save(y_dir + f'buffer_y_mosaics_{dist}px.npy', buffers_y_mosaics[dist])
            
        # Buffer dos Mosaicos de Predição   
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_mosaics_{dist}px.npy')):
            buffers_pred_mosaics[dist] = buffer_patches(pred_mosaics, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_mosaics_{dist}px.npy', buffers_pred_mosaics[dist])
    
    
    # Lê buffers de Mosaicos
    for dist in dist_buffers:
        buffers_y_mosaics[dist] = np.load(y_dir + f'buffer_y_mosaics_{dist}px.npy')
        buffers_pred_mosaics[dist] = np.load(output_dir + f'buffer_pred_mosaics_{dist}px.npy')  
        
        
    # Avaliação da Qualidade para os Mosaicos de Teste
    # Relaxed Metrics for mosaics test
    relaxed_precision_mosaics, relaxed_recall_mosaics, relaxed_f1score_mosaics = {}, {}, {}
      
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
            relaxed_precision_mosaics[dist], relaxed_recall_mosaics[dist], relaxed_f1score_mosaics[dist] = compute_relaxed_metrics(y_mosaics, 
                                                                                                            pred_mosaics, buffers_y_mosaics[dist],
                                                                                                            buffers_pred_mosaics[dist], 
                                                                                                            nome_conjunto = 'Mosaicos de Teste', 
                                                                                                            print_file=f)
            
    # Libera memória
    y_mosaics = None
    pred_mosaics = None
    
    buffers_y_mosaics = None
    buffers_pred_mosaics = None
    
    gc.collect()
    
    
    # Gera Diferença Referente a Novas Estradas e
    # Avalia Novas Estradas
    if avalia_diff:
        if not os.path.exists(os.path.join(y_dir, 'y_tiles_diff.npy')) or \
           not os.path.exists(os.path.join(output_dir, 'pred_tiles_diff.npy')):
                        
            # Lista de Tiles de Referência de Antes
            test_labels_tiles_before_dir = Path(r'tiles/masks/2016/test')
            test_labels_tiles_before = list(test_labels_tiles_before_dir.glob('*.tif'))
        
            # Lista de Tiles Preditos (referente a Depois)
            test_labels_tiles_predafter = list(Path(output_dir).glob('outmosaic*.tif'))
        
            # Extrai Diferenca entre Predição e Referência de Antes para
            # Computar Novas Estradas
            for tile_before, tile_after in zip(test_labels_tiles_before, test_labels_tiles_predafter):
                print(f'Extraindo Diferença entre {tile_after.name} e {tile_before.name}')
                suffix_extension = Path(tile_after).name.replace('outmosaic_', '', 1)
                out_raster_path = Path(output_dir) / f"diffnewroad_{suffix_extension}"
                extract_difference_reftiles(str(tile_before), str(tile_after), str(out_raster_path), buffer_px=3)
                
            # Lista de caminhos dos rótulos dos Tiles de Diferança de Teste
            test_labels_tiles_diff_dir = Path(r'tiles/masks/Diff/test')
            test_labels_tiles_diff = list(test_labels_tiles_diff_dir.glob('*.tif'))
            
            # Lista e Array da referência para os Tiles de Diferença
            y_tiles_diff = [gdal.Open(str(tile_diff)).ReadAsArray() for tile_diff in test_labels_tiles_diff]
            y_tiles_diff = stack_uneven(y_tiles_diff)[..., np.newaxis]
            
            # Lista e Array da predição da Diferença
            test_labels_tiles_preddiff = list(Path(output_dir).glob('diffnewroad*.tif'))
            pred_tiles_diff = [gdal.Open(str(tile_preddiff)).ReadAsArray() for tile_preddiff in test_labels_tiles_preddiff]
            pred_tiles_diff = stack_uneven(pred_tiles_diff)[..., np.newaxis]
            
            # Salva Arrays dos Rótulos e Predições das Diferenças
            salva_arrays(y_dir, y_tiles_diff=y_tiles_diff)
            salva_arrays(output_dir, pred_tiles_diff=pred_tiles_diff)
            
            
        # Lê Arrays dos Rótulos e Predições das Diferenças
        y_tiles_diff = np.load(y_dir + 'y_tiles_diff.npy')
        pred_tiles_diff = np.load(output_dir + 'pred_tiles_diff.npy')
        
        # Buffer dos Tiles de Referência e Predição da Diferença
        buffers_y_tiles_diff = {}
        buffers_pred_tiles_diff = {}
        
        for dist in dist_buffers:
            # Buffer da referência dos Tiles de Diferença
            if not os.path.exists(os.path.join(y_dir, f'buffer_y_tiles_diff_{dist}px.npy')):
                buffers_y_tiles_diff[dist] = buffer_patches(y_tiles_diff, dist_cells=dist)
                np.save(y_dir + f'buffer_y_tiles_diff_{dist}px.npy', buffers_y_tiles_diff[dist])
                
            # Buffer da predição da Diferença   
            if not os.path.exists(os.path.join(output_dir, f'buffer_pred_tiles_diff_{dist}px.npy')):
                buffers_pred_tiles_diff[dist] = buffer_patches(pred_tiles_diff, dist_cells=dist)
                np.save(output_dir + f'buffer_pred_tiles_diff_{dist}px.npy', buffers_pred_tiles_diff[dist])
                
        
        # Lê buffers das Diferenças
        for dist in dist_buffers:
            buffers_y_tiles_diff[dist] = np.load(y_dir + f'buffer_y_tiles_diff_{dist}px.npy')
            buffers_pred_tiles_diff[dist] = np.load(output_dir + f'buffer_pred_tiles_diff_{dist}px.npy')
            
        # Avaliação da Qualidade para a Diferença
        # Relaxed Metrics for difference tiles
        relaxed_precision_diff, relaxed_recall_diff, relaxed_f1score_diff = {}, {}, {}
          
        for dist in dist_buffers:
            with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
                relaxed_precision_diff[dist], relaxed_recall_diff[dist], relaxed_f1score_diff[dist] = compute_relaxed_metrics(y_tiles_diff, 
                                                                                                            pred_tiles_diff, buffers_y_tiles_diff[dist],
                                                                                                            buffers_pred_tiles_diff[dist], 
                                                                                                            nome_conjunto = 'Mosaicos de Diferenca', 
                                                                                                            print_file=f)
                
        # Libera memória
        y_tiles_diff = None
        pred_tiles_diff = None
        
        buffers_y_tiles_diff = None
        buffers_pred_tiles_diff = None
        
        gc.collect()
            
    
    # Save and Return dictionary with the values of precision, recall and F1-Score for all the groups (Train, Validation, Test, Mosaics of Test)
    if not avalia_train:
        relaxed_precision_train = None
        relaxed_recall_train = None
        relaxed_f1score_train = None

    if not avalia_diff:
        relaxed_precision_diff = None
        relaxed_recall_diff = None
        relaxed_f1score_diff = None           
        
    dict_results = {
        'relaxed_precision_train': relaxed_precision_train, 'relaxed_recall_train': relaxed_recall_train, 'relaxed_f1score_train': relaxed_f1score_train,
        'relaxed_precision_valid': relaxed_precision_valid, 'relaxed_recall_valid': relaxed_recall_valid, 'relaxed_f1score_valid': relaxed_f1score_valid,
        'relaxed_precision_test': relaxed_precision_test, 'relaxed_recall_test': relaxed_recall_test, 'relaxed_f1score_test': relaxed_f1score_test,
        'relaxed_precision_mosaics': relaxed_precision_mosaics, 'relaxed_recall_mosaics': relaxed_recall_mosaics, 'relaxed_f1score_mosaics': relaxed_f1score_mosaics,
        'relaxed_precision_diff': relaxed_precision_diff, 'relaxed_recall_diff': relaxed_recall_diff, 'relaxed_f1score_diff': relaxed_f1score_diff
        }
    
    with open(output_dir + 'resultados_metricas' + '.pickle', "wb") as fp: # Salva dicionário com métricas do resultado do experimento
        pickle.dump(dict_results, fp)
    
    return dict_results









# Avalia um ensemble de modelos segundo conjuntos de treino, validação, teste e mosaicos de teste
# Retorna as métricas de precisão, recall e f1-score relaxados
# Além disso constroi os mosaicos de resultado 
# Etapa 1 se refere ao processamento no qual a entrada do pós-processamento é a predição do modelo
# Etapa 2 ao processamento no qual a entrada do pós-processamento é a imagem original mais a predição do modelo
# Etapa 3 ao processamento no qual a entrada do pós-processamento é a imagem original com blur gaussiano mais a predição desses dados ruidosos com o modelo
# Etapa 4 ao processamento no qual a entrada do pós-processamento é a imagem original mais a predição dela aplicando um blur gaussiano em apenas alguns subpatches
# Etapa 5 se refere ao pós-processamento   
def avalia_modelo_ensemble(input_dir: str, output_dir: str, metric_name = 'F1-Score', 
                           etapa=3, dist_buffers = [3], std_blur = 0.4, n_members=5):
    metric_name = metric_name
    etapa = etapa
    dist_buffers = dist_buffers
    std_blur = std_blur
    
    # Lê arrays referentes aos patches de treino e validação
    x_train = np.load(input_dir + 'x_train.npy')
    y_train = np.load(input_dir + 'y_train.npy')
    x_valid = np.load(input_dir + 'x_valid.npy')
    y_valid = np.load(input_dir + 'y_valid.npy')
    
    # Copia y_train e y_valid para diretório de saída, pois eles serão usados depois como entrada (rótulos)
    # para o pós-processamento
    if etapa==1 or etapa==2 or etapa==3:
        shutil.copy(input_dir + 'y_train.npy', output_dir + 'y_train.npy')
        shutil.copy(input_dir + 'y_valid.npy', output_dir + 'y_valid.npy')
    
    
    # Nome base do modelo salvo e número de membros do ensemble    
    best_model_filename = 'best_model'
    n_members = n_members
    
    # Show and save history
    for i in range(n_members):
        with open(output_dir + 'history_pickle_' + best_model_filename + '_' + str(i+1) + '.pickle', "rb") as fp:   
            history = pickle.load(fp)
            
        # Show and save history
        show_graph_loss_accuracy(np.asarray(history), 1, metric_name = metric_name, save=True, save_path=output_dir,
                                 save_name='plotagem' + '_' + str(i+1) + '.png')
    
    # Load model
    model_list = []
    
    for i in range(n_members):
        model = load_model(output_dir + best_model_filename + '_' + str(i+1) + '.h5', compile=False)
        model_list.append(model)
        
    # Test the model over training and validation data (outputs result if at least one of the files does not exist)
    if not os.path.exists(os.path.join(output_dir, 'pred_train_ensemble.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'prob_train_ensemble.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'pred_valid_ensemble.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'prob_valid_ensemble.npy')):
        pred_train_ensemble, prob_train_ensemble = Test_Step_Ensemble(model_list, x_train, 2)
        pred_valid_ensemble, prob_valid_ensemble = Test_Step_Ensemble(model_list, x_valid, 2)
            
        # Converte para tipos que ocupam menos espaço
        pred_train_ensemble = pred_train_ensemble.astype(np.uint8)
        prob_train_ensemble = prob_train_ensemble.astype(np.float16)
        pred_valid_ensemble = pred_valid_ensemble.astype(np.uint8)
        prob_valid_ensemble = prob_valid_ensemble.astype(np.float16)
            
        # Salva arrays de predição do Treinamento e Validação   
        salva_arrays(output_dir, pred_train_ensemble=pred_train_ensemble, prob_train_ensemble=prob_train_ensemble, 
                     pred_valid_ensemble=pred_valid_ensemble, prob_valid_ensemble=prob_valid_ensemble)
       
        
    # Lê arrays de predição do Treinamento e Validação (para o caso deles já terem sido gerados)  
    pred_train_ensemble = np.load(output_dir + 'pred_train_ensemble.npy')
    prob_train_ensemble = np.load(output_dir + 'prob_train_ensemble.npy')
    pred_valid_ensemble = np.load(output_dir + 'pred_valid_ensemble.npy')
    prob_valid_ensemble = np.load(output_dir + 'prob_valid_ensemble.npy')
    
    # Calcula e salva predição a partir da probabilidade média do ensemble
    if not os.path.exists(os.path.join(output_dir, 'pred_train.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'pred_valid.npy')):
        pred_train = calcula_pred_from_prob_ensemble_mean(prob_train_ensemble)
        pred_valid = calcula_pred_from_prob_ensemble_mean(prob_valid_ensemble)
        
        pred_train = pred_train.astype(np.uint8)
        pred_valid = pred_valid.astype(np.uint8)
        
        salva_arrays(output_dir, pred_train=pred_train, pred_valid=pred_valid)
        
    
    # Lê arrays de predição do Treinamento e Validação (para o caso deles já terem sido gerados)  
    pred_train = np.load(output_dir + 'pred_train.npy')
    pred_valid = np.load(output_dir + 'pred_valid.npy')
        
        
    
    # Antigo: Copia predições pred_train e pred_valid com o nome de x_train e x_valid no diretório de saída,
    # Antigo: pois serão usados como dados de entrada para o pós-processamento
    # Faz concatenação de x_train com pred_train e salva no diretório de saída, com o nome de x_train,
    # para ser usado como entrada para o pós-processamento. Faz procedimento semelhante com x_valid
    if etapa == 1:
        if not os.path.exists(os.path.join(output_dir, 'x_train.npy')) or \
            not os.path.exists(os.path.join(output_dir, 'x_valid.npy')):
            shutil.copy(output_dir + 'pred_train.npy', output_dir + 'x_train.npy')
            shutil.copy(output_dir + 'pred_valid.npy', output_dir + 'x_valid.npy')
    
    if etapa == 2:
        if not os.path.exists(os.path.join(output_dir, 'x_train.npy')) or \
            not os.path.exists(os.path.join(output_dir, 'x_valid.npy')):
            x_train_new = np.concatenate((x_train, pred_train), axis=-1)
            x_valid_new = np.concatenate((x_valid, pred_valid), axis=-1)
            salva_arrays(output_dir, x_train=x_train_new, x_valid=x_valid_new)
        
    if etapa == 3:
        if not os.path.exists(os.path.join(output_dir, 'x_train.npy')) or \
            not os.path.exists(os.path.join(output_dir, 'x_valid.npy')):
            print('Fazendo Blur nas imagens de treino e validação e gerando as predições')
            x_train_blur = gaussian_filter(x_train.astype(np.float32), sigma=(0 ,std_blur, std_blur, 0)).astype(np.float16)
            x_valid_blur = gaussian_filter(x_valid.astype(np.float32), sigma=(0 ,std_blur, std_blur, 0)).astype(np.float16)
            _, prob_train_ensemble_blur = Test_Step_Ensemble(model, x_train_blur, 2)
            _, prob_valid_ensemble_blur = Test_Step_Ensemble(model, x_valid_blur, 2)
            pred_train_blur = calcula_pred_from_prob_ensemble_mean(prob_train_ensemble_blur)
            pred_valid_blur = calcula_pred_from_prob_ensemble_mean(prob_valid_ensemble_blur)
            x_train_new = np.concatenate((x_train_blur, pred_train_blur), axis=-1).astype(np.float16)
            x_valid_new = np.concatenate((x_valid_blur, pred_valid_blur), axis=-1).astype(np.float16)
            salva_arrays(output_dir, x_train=x_train_new, x_valid=x_valid_new)
            


        
        
    # Faz os Buffers necessários, para treino e validação, nas imagens 
    
    # Precisão Relaxada - Buffer na imagem de rótulos
    # Sensibilidade Relaxada - Buffer na imagem extraída
    # F1-Score Relaxado - É obtido através da Precisão e Sensibilidade Relaxadas
    
    buffers_y_train = {}
    buffers_y_valid = {}
    buffers_pred_train = {}
    buffers_pred_valid = {}
    
    for dist in dist_buffers:
        # Buffers para Precisão Relaxada
        if not os.path.exists(os.path.join(input_dir, f'buffer_y_train_{dist}px.npy')): 
            buffers_y_train[dist] = buffer_patches(y_train, dist_cells=dist)
            np.save(input_dir + f'buffer_y_train_{dist}px.npy', buffers_y_train[dist])
            
        if not os.path.exists(os.path.join(input_dir, f'buffer_y_valid_{dist}px.npy')): 
            buffers_y_valid[dist] = buffer_patches(y_valid, dist_cells=dist)
            np.save(input_dir + f'buffer_y_valid_{dist}px.npy', buffers_y_valid[dist])
        
        # Buffers para Sensibilidade Relaxada
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_train_{dist}px.npy')):
            buffers_pred_train[dist] = buffer_patches(pred_train, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_train_{dist}px.npy', buffers_pred_train[dist])
            
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_valid_{dist}px.npy')): 
            buffers_pred_valid[dist] = buffer_patches(pred_valid, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_valid_{dist}px.npy', buffers_pred_valid[dist])
    
    
    # Lê buffers de arrays de predição do Treinamento e Validação
    for dist in dist_buffers:
        buffers_y_train[dist] = np.load(input_dir + f'buffer_y_train_{dist}px.npy')
        buffers_y_valid[dist] = np.load(input_dir + f'buffer_y_valid_{dist}px.npy')
        buffers_pred_train[dist] = np.load(output_dir + f'buffer_pred_train_{dist}px.npy')
        buffers_pred_valid[dist] = np.load(output_dir + f'buffer_pred_valid_{dist}px.npy')
    
    
    # Relaxed Metrics for training and validation
    relaxed_precision_train, relaxed_recall_train, relaxed_f1score_train = {}, {}, {}
    relaxed_precision_valid, relaxed_recall_valid, relaxed_f1score_valid = {}, {}, {}
    
    
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
            relaxed_precision_train[dist], relaxed_recall_train[dist], relaxed_f1score_train[dist] = compute_relaxed_metrics(y_train, 
                                                                                                       pred_train, buffers_y_train[dist],
                                                                                                       buffers_pred_train[dist], 
                                                                                                       nome_conjunto = 'Treino', 
                                                                                                       print_file=f)        
            # Relaxed Metrics for validation
            relaxed_precision_valid[dist], relaxed_recall_valid[dist], relaxed_f1score_valid[dist] = compute_relaxed_metrics(y_valid, 
                                                                                                           pred_valid, buffers_y_valid[dist],
                                                                                                           buffers_pred_valid[dist],
                                                                                                           nome_conjunto = 'Validação',
                                                                                                           print_file=f) 
        
        
    # Lê arrays referentes aos patches de teste
    x_test = np.load(input_dir + 'x_test.npy')
    y_test = np.load(input_dir + 'y_test.npy')
    
    
    # Copia y_test para diretório de saída, pois ele será usado como entrada (rótulo) para o pós-processamento
    if etapa==1 or etapa==2 or etapa==3:
        shutil.copy(input_dir + 'y_test.npy', output_dir + 'y_test.npy')
        
    
    # Test the model over test data (outputs result if at least one of the files does not exist)
    if not os.path.exists(os.path.join(output_dir, 'pred_test_ensemble.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'prob_test_ensemble.npy')):
        pred_test_ensemble, prob_test_ensemble = Test_Step_Ensemble(model_list, x_test, 2)
        
        # Converte para tipos que ocupam menos espaço
        pred_test_ensemble = pred_test_ensemble.astype(np.uint8)
        prob_test_ensemble = prob_test_ensemble.astype(np.float16)
        
        # Salva arrays de predição do Teste. Arquivos da Predição (pred) são salvos na pasta de arquivos de saída (resultados_dir)
        salva_arrays(output_dir, pred_test_ensemble=pred_test_ensemble, prob_test_ensemble=prob_test_ensemble)
        
        
    # Lê arrays de predição do Teste
    pred_test_ensemble = np.load(output_dir + 'pred_test_ensemble.npy')
    prob_test_ensemble = np.load(output_dir + 'prob_test_ensemble.npy')
    
    
    # Calcula e salva predição a partir da probabilidade média do ensemble
    if not os.path.exists(os.path.join(output_dir, 'pred_test.npy')):
        pred_test = calcula_pred_from_prob_ensemble_mean(prob_test_ensemble)
        
        pred_test = pred_test.astype(np.uint8)
        
        salva_arrays(output_dir, pred_test=pred_test)
        
    
    # Lê arrays de predição do Teste
    pred_test = np.load(output_dir + 'pred_test.npy')
    
    # Antigo: Copia predição pred_test com o nome de x_test no diretório de saída,
    # Antigo: pois será usados como dado de entrada no pós-processamento
    # Faz concatenação de x_test com pred_test e salva no diretório de saída, com o nome de x_test,
    # para ser usado como entrada para o pós-processamento.
    if etapa == 1:
        if not os.path.exists(os.path.join(output_dir, 'x_test.npy')):
            shutil.copy(output_dir + 'pred_test.npy', output_dir + 'x_test.npy')
    
    if etapa == 2:
        if not os.path.exists(os.path.join(output_dir, 'x_test.npy')):
            x_test_new = np.concatenate((x_test, pred_test), axis=-1)
            salva_arrays(output_dir, x_test=x_test_new)
        
    if etapa == 3:
        if not os.path.exists(os.path.join(output_dir, 'x_test.npy')):
            print('Fazendo Blur nas imagens de teste e gerando as predições')
            x_test_blur = gaussian_filter(x_test.astype(np.float32), sigma=(0 ,std_blur, std_blur, 0)).astype(np.float16)
            _, prob_test_ensemble_blur = Test_Step_Ensemble(model, x_test_blur, 2)
            pred_test_blur = calcula_pred_from_prob_ensemble_mean(prob_test_ensemble_blur)
            x_test_new = np.concatenate((x_test_blur, pred_test_blur), axis=-1).astype(np.float16)
            salva_arrays(output_dir, x_test=x_test_new)
    
    
    # Faz os Buffers necessários, para teste, nas imagens
    buffers_y_test = {}
    buffers_pred_test = {}
    
    for dist in dist_buffers:
        # Buffer para Precisão Relaxada
        if not os.path.exists(os.path.join(input_dir, f'buffer_y_test_{dist}px.npy')):
            buffers_y_test[dist] = buffer_patches(y_test, dist_cells=dist)
            np.save(input_dir + f'buffer_y_test_{dist}px.npy', buffers_y_test[dist])
            
        # Buffer para Sensibilidade Relaxada
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_test_{dist}px.npy')):
            buffers_pred_test[dist] = buffer_patches(pred_test, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_test_{dist}px.npy', buffers_pred_test[dist])
            
            
    # Lê buffers de arrays de predição do Teste
    for dist in dist_buffers:
        buffers_y_test[dist] = np.load(input_dir + f'buffer_y_test_{dist}px.npy')
        buffers_pred_test[dist] = np.load(output_dir + f'buffer_pred_test_{dist}px.npy')
    
    # Relaxed Metrics for test
    relaxed_precision_test, relaxed_recall_test, relaxed_f1score_test = {}, {}, {}
    
    
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
            relaxed_precision_test[dist], relaxed_recall_test[dist], relaxed_f1score_test[dist] = compute_relaxed_metrics(y_test, 
                                                                                                       pred_test, buffers_y_test[dist],
                                                                                                       buffers_pred_test[dist], 
                                                                                                       nome_conjunto = 'Teste', 
                                                                                                       print_file=f)
        
    
    # Stride e Dimensões do Tile
    patch_test_stride = 96 # tem que estabelecer de forma manual de acordo com o stride usado para extrair os patches
    labels_test_shape = (1408, 1280) # também tem que estabelecer de forma manual de acordo com as dimensões da imagem de referência
    n_test_tiles = 10 # Número de tiles de teste
    
    # Pasta com os tiles de teste para pegar informações de georreferência
    test_labels_tiles_dir = r'new_teste_tiles\masks\test'
    labels_paths = [os.path.join(test_labels_tiles_dir, arq) for arq in os.listdir(test_labels_tiles_dir)]
    
    
    
    # Gera mosaicos e lista com os mosaicos previstos
    pred_test_mosaic_list = gera_mosaicos(output_dir, pred_test, labels_paths, 
                                          patch_test_stride=patch_test_stride,
                                          labels_test_shape=labels_test_shape,
                                          n_test_tiles=n_test_tiles, is_float=False)
    
    
    # Lista e Array dos Mosaicos de Referência
    y_mosaics_list = [gdal.Open(y_mosaic_path).ReadAsArray() for y_mosaic_path in labels_paths]
    y_mosaics = np.array(y_mosaics_list)[..., np.newaxis]
    
    # Array dos Mosaicos de Predição 
    pred_mosaics = np.array(pred_test_mosaic_list)[..., np.newaxis]
    pred_mosaics = pred_mosaics.astype(np.uint8)
    
    # Salva Array dos Mosaicos de Predição
    if not os.path.exists(os.path.join(input_dir, 'y_mosaics.npy')): salva_arrays(input_dir, y_mosaics=y_mosaics)
    if not os.path.exists(os.path.join(output_dir, 'pred_mosaics.npy')): salva_arrays(output_dir, pred_mosaics=pred_mosaics)
    
    # Lê Mosaicos 
    y_mosaics = np.load(input_dir + 'y_mosaics.npy')
    pred_mosaics = np.load(output_dir + 'pred_mosaics.npy')
    
    
    # Buffer dos Mosaicos de Referência e Predição
    buffers_y_mosaics = {}
    buffers_pred_mosaics = {}
    
    for dist in dist_buffers:
        # Buffer dos Mosaicos de Referência
        if not os.path.exists(os.path.join(input_dir, f'buffers_y_mosaics_{dist}px.npy')):
            buffers_y_mosaics[dist] = buffer_patches(y_mosaics, dist_cells=dist)
            np.save(input_dir + f'buffer_y_mosaics_{dist}px.npy', buffers_y_mosaics[dist])
            
        # Buffer dos Mosaicos de Predição   
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_mosaics_{dist}px.npy')):
            buffers_pred_mosaics[dist] = buffer_patches(pred_mosaics, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_mosaics_{dist}px.npy', buffers_pred_mosaics[dist])
    
    
    # Lê buffers de Mosaicos
    for dist in dist_buffers:
        buffers_y_mosaics[dist] = np.load(input_dir + f'buffer_y_mosaics_{dist}px.npy')
        buffers_pred_mosaics[dist] = np.load(output_dir + f'buffer_pred_mosaics_{dist}px.npy')  
    
    # Avaliação da Qualidade para os Mosaicos de Teste
    # Relaxed Metrics for mosaics test
    relaxed_precision_mosaics, relaxed_recall_mosaics, relaxed_f1score_mosaics = {}, {}, {}
      
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}px.txt', 'a') as f:
            relaxed_precision_mosaics[dist], relaxed_recall_mosaics[dist], relaxed_f1score_mosaics[dist] = compute_relaxed_metrics(y_mosaics, 
                                                                                                       pred_mosaics, buffers_y_mosaics[dist],
                                                                                                       buffers_pred_mosaics[dist], 
                                                                                                       nome_conjunto = 'Mosaicos de Teste', 
                                                                                                       print_file=f)
            
    
    # Save and Return dictionary with the values of precision, recall and F1-Score for all the groups (Train, Validation, Test, Mosaics of Test)
    dict_results = {
        'relaxed_precision_train': relaxed_precision_train, 'relaxed_recall_train': relaxed_recall_train, 'relaxed_f1score_train': relaxed_f1score_train,
        'relaxed_precision_valid': relaxed_precision_valid, 'relaxed_recall_valid': relaxed_recall_valid, 'relaxed_f1score_valid': relaxed_f1score_valid,
        'relaxed_precision_test': relaxed_precision_test, 'relaxed_recall_test': relaxed_recall_test, 'relaxed_f1score_test': relaxed_f1score_test,
        'relaxed_precision_mosaics': relaxed_precision_mosaics, 'relaxed_recall_mosaics': relaxed_recall_mosaics, 'relaxed_f1score_mosaics': relaxed_f1score_mosaics             
        }
    
    with open(output_dir + 'resultados_metricas' + '.pickle', "wb") as fp: # Salva dicionário com métricas do resultado do experimento
        pickle.dump(dict_results, fp)
    
    return dict_results












# Função para cálculo da entropia preditiva para um array de probabilidade com várias predições
def calcula_predictive_entropy(prob_array):
    # Calcula probabilidade média    
    prob_mean = np.mean(prob_array, axis=0)
    
    # Calcula Entropia Preditiva
    pred_entropy = np.zeros(prob_mean.shape[0:-1] + (1,)) # Inicia com zeros
    
    K = prob_mean.shape[-1] # número de classes
    epsilon = 1e-7 # usado para evitar log 0
    
    '''
    # Com log base 2 porque tem 2 classes
    for k in range(K):
        pred_entropy_valid = pred_entropy_valid + prob_valid_mean[..., k:k+1] * np.log2(prob_valid_mean[..., k:k+1] + epsilon) 
        
    for k in range(K):
        pred_entropy_train = pred_entropy_train + prob_train_mean[..., k:k+1] * np.log2(prob_train_mean[..., k:k+1] + epsilon) 
    '''    

    # Calculando com log base e e depois escalonando para entre 0 e 1
    for k in range(K):
        pred_entropy = pred_entropy + prob_mean[..., k:k+1] * np.log(prob_mean[..., k:k+1] + epsilon) 
        
    pred_entropy = - pred_entropy / np.log(K) # Escalona entre 0 e 1, já que o máximo da entropia é log(K),
                                              # onde K é o número de classes
                                              
    pred_entropy = np.clip(pred_entropy, 0, 1) # Clipa valores entre 0 e 1

    return pred_entropy  

# Calcula predição a partir da probabilidade média do ensemble
def calcula_pred_from_prob_ensemble_mean(prob_array):
    # Calcula probabilidade média    
    prob_mean = np.mean(prob_array, axis=0)

    # Calcula predição (é a classe que, entre as 2 probabilidades, tem probabilidade maior)
    predicted_classes = np.argmax(prob_mean, axis=-1)[..., np.newaxis] 
    
    return predicted_classes



# pred_array é o array de predição e labels_paths é uma lista com o nome dos caminhos dos tiles
# que deram origem aos patches e que serão usados para informações de georreferência
# patch_test_stride é o stride com o qual foram extraídos os patches, necessário para a reconstrução dos tiles
# labels_test_shape é a dimensão de cada tile, também necessário para suas reconstruções
# len_tiles_test é a quantidade de patches por tile
def gera_mosaicos(output_dir, pred_array, labels_paths, prefix='outmosaic', patch_test_stride=96, labels_test_shape=(1408, 1280), len_tiles_test=[], is_float=False):
    # Stride e Dimensões do Tile
    patch_test_stride = patch_test_stride # tem que estabelecer de forma manual de acordo com o stride usado para extrair os patches
    labels_test_shape = labels_test_shape # também tem que estabelecer de forma manual de acordo com as dimensões da imagem de referência
    
    # Lista com os mosaicos previstos
    pred_test_mosaic_list = []
    
    # n_test_tiles = n_test_tiles
    n_test_tiles = len(len_tiles_test)
    # n_patches_tile = int(pred_array.shape[0]/n_test_tiles) # Quantidade de patches por tile. Supõe tiles com igual número de patches
    n = len(pred_array) # Quantidade de patches totais, contando todos os tiles, é igual ao comprimento 
    
    i = 0 # Índice onde o tile começa
    i_mosaic = 0 # Índice que se refere ao número do mosaico (primeiro, segundo, terceiro, ...)
    
    while i < n:
        print('Making Mosaic {}/{}'.format(i_mosaic+1, n_test_tiles ))
        
        pred_test_mosaic = unpatch_reference(pred_array[i:i+len_tiles_test[i_mosaic], ..., 0], patch_test_stride, labels_test_shape[i_mosaic], border_patches=True)
        
        pred_test_mosaic_list.append(pred_test_mosaic) 
        
        # Incremento
        i = i+len_tiles_test[i_mosaic]
        i_mosaic += 1  
        
        
    # Salva mosaicos
    labels_paths = labels_paths
    output_dir = output_dir
    
    for i in range(len(pred_test_mosaic_list)):
        pred_test_mosaic = pred_test_mosaic_list[i]
        labels_path = labels_paths[i]
        
        filename_wo_ext = labels_path.split(os.path.sep)[-1].split('.tif')[0]
        '''
        tile_line = int(filename_wo_ext.split('_')[1])
        tile_col = int(filename_wo_ext.split('_')[2])
        
        out_mosaic_name = prefix + '_' + str(tile_line) + '_' + str(tile_col) + r'.tif'
        '''
        out_mosaic_name = prefix + '_' + filename_wo_ext + r'.tif'
        out_mosaic_path = os.path.join(output_dir, out_mosaic_name)
        
        if is_float:
            save_raster_reference(labels_path, out_mosaic_path, pred_test_mosaic, is_float=True)
        else:
            save_raster_reference(labels_path, out_mosaic_path, pred_test_mosaic, is_float=False)
    
            
            
    # Retorna lista de mosaicos
    return pred_test_mosaic_list
            



def patches_to_images_tf(
    patches: np.ndarray, image_shape: tuple,
    overlap: float = 0.5, stitch_type='average', indices=None) -> np.ndarray:
    """Reconstructs images from patches.

    Args:
        patches (ndarray): Array with batch of patches to convert to batch of images.
            [batch_size, #patch_y, #patch_x, patch_size_y, patch_size_x, n_channels]
        image_shape (Tuple): Shape of output image. (y, x, n_channels) or (y, x)
        overlap (float, optional): Overlap factor between patches. Defaults to 0.5.
        stitch_type (str, optional): Type of stitching to use. Defaults to 'average'.
            Options: 'average', 'replace'.
        indices (ndarray, optional): Indices of patches in image. Defaults to None.
            If provided, indices are used to stitch patches together and not recomputed
            to save time. Has same shape as patches shape but with added index axis (last).
    Returns:
        images (ndarray): Reconstructed batch of images from batch of patches.

    """
    assert len(image_shape) == 3, 'image_shape should have 3 dimensions, namely: ' \
        '(#image_y, #image_x, (n_channels))'
    assert len(patches.shape) == 6 , 'patches should have 6 dimensions, namely: ' \
        '[batch, #patch_y, #patch_x, patch_size_y, patch_size_x, n_channels]'
    assert overlap >= 0 and overlap < 1, 'overlap should be between 0 and 1'
    assert stitch_type in ['average', 'replace'], 'stitch_type should be either ' \
        '"average" or "replace"'

    batch_size, n_patches_y, n_patches_x, *patch_shape = patches.shape
    n_channels = image_shape[-1]
    dtype = patches.dtype

    assert len(patch_shape) == len(image_shape)

    # Kernel for counting overlaps
    if stitch_type == 'average':
        kernel_ones = tf.ones((batch_size, n_patches_y, n_patches_x, *patch_shape), dtype=tf.int32)
        mask = tf.zeros((batch_size, *image_shape), dtype=tf.int32)

    if indices is None:
        if overlap:
            nonoverlap = np.array([1 - overlap, 1 - overlap, 1 / patch_shape[-1]])
            stride = (np.array(patch_shape) * nonoverlap).astype(int)
        else:
            stride = (np.array(patch_shape)).astype(int)

        channel_idx = tf.reshape(tf.range(n_channels), (1, 1, 1, 1, 1, n_channels, 1))
        channel_idx = (tf.ones((batch_size, n_patches_y, n_patches_x, *patch_shape, 1), dtype=tf.int32) * channel_idx)

        batch_idx = tf.reshape(tf.range(batch_size), (batch_size, 1, 1, 1, 1, 1, 1))
        batch_idx = (tf.ones((batch_size, n_patches_y, n_patches_x, *patch_shape, 1), dtype=tf.int32) * batch_idx)

        # TODO: create indices without looping possibly
        indices = []
        for j in range(n_patches_y):
            for i in range(n_patches_x):
                # Make indices from meshgrid
                _indices = tf.meshgrid(
                    tf.range(stride[0] * j, # row start
                            patch_shape[0] + stride[0] * j), # row end
                    tf.range(stride[1] * i, # col_start
                            patch_shape[1] + stride[1] * i), indexing='ij') # col_end

                _indices = tf.stack(_indices, axis=-1)
                indices.append(_indices)

        indices = tf.reshape(tf.stack(indices, axis=0), (n_patches_y, n_patches_x, *patch_shape[:2], 2))

        indices = tf.repeat(indices[tf.newaxis, ...], batch_size, axis=0)
        indices = tf.repeat(indices[..., tf.newaxis, :], n_channels, axis=-2)

        indices = tf.concat([batch_idx, indices, channel_idx], axis=-1)

    # create output image tensor
    images = tf.zeros([batch_size, *image_shape], dtype=dtype)

    # Add sliced image to recovered image indices
    if stitch_type == 'replace':
        images = tf.tensor_scatter_nd_update(images, indices, patches)
    else:
        images = tf.tensor_scatter_nd_add(images, indices, patches)
        mask = tf.tensor_scatter_nd_add(mask, indices, kernel_ones)
        images = tf.cast(images, tf.float32) / tf.cast(mask, tf.float32)

    return images, indices





def mode(ndarray, axis=0):
    # Check inputs
    ndarray = np.asarray(ndarray)
    ndim = ndarray.ndim
    if ndarray.size == 1:
        return (ndarray[0], 1)
    elif ndarray.size == 0:
        raise Exception('Cannot compute mode on empty array')
    try:
        axis = range(ndarray.ndim)[axis]
    except:
        raise Exception('Axis "{}" incompatible with the {}-dimension array'.format(axis, ndim))

    # If array is 1-D and numpy version is > 1.9 numpy.unique will suffice
    if all([ndim == 1,
            int(np.__version__.split('.')[0]) >= 1,
            int(np.__version__.split('.')[1]) >= 9]):
        modals, counts = np.unique(ndarray, return_counts=True)
        index = np.argmax(counts)
        return modals[index], counts[index]

    # Sort array
    sort = np.sort(ndarray, axis=axis)
    # Create array to transpose along the axis and get padding shape
    transpose = np.roll(np.arange(ndim)[::-1], axis)
    shape = list(sort.shape)
    shape[axis] = 1
    # Create a boolean array along strides of unique values
    strides = np.concatenate([np.zeros(shape=shape, dtype='bool'),
                                 np.diff(sort, axis=axis) == 0,
                                 np.zeros(shape=shape, dtype='bool')],
                                axis=axis).transpose(transpose).ravel()
    # Count the stride lengths
    counts = np.cumsum(strides)
    counts[~strides] = np.concatenate([[0], np.diff(counts[~strides])])
    counts[strides] = 0
    # Get shape of padded counts and slice to return to the original shape
    shape = np.array(sort.shape)
    shape[axis] += 1
    shape = shape[transpose]
    slices = [slice(None)] * ndim
    slices[axis] = slice(1, None)
    # Reshape and compute final counts
    counts = counts.reshape(shape).transpose(transpose)[slices] + 1

    # Find maximum counts and return modals/counts
    slices = [slice(None, i) for i in sort.shape]
    del slices[axis]
    index = np.ogrid[slices]
    index.insert(axis, np.argmax(counts, axis=axis))
    return sort[index], counts[index]








def blur_x_patches(x_train, y_train, dim, k, blur, model):
    '''
    Faz o blur, em cada patch, em k subpaches. Preferencialmente naqueles que contêm pelo menos um pixel de estrada.
    Caso não houver subpatches suficientes subpatches com pelos um pixel de estrada, então faz o blur naqueles que não 
    tem estrada mesmo.

    Parameters
    ----------
    x_train : TYPE
        Numpy array (patches, height, width, channels).
    y_train : TYPE
        Numpy array (patches, height, width, channels).
    dim : TYPE
        Tamanho do subpatch
    k : TYPE
        Quantos subpatches aplica o blur.
    blur : TYPE
        Desvio Padrão do blur.

    Returns
    -------
    x_train_blur : TYPE
        x_train com blur aplicado.
    pred_train_blur : TYPE
        predições de x_train com blur aplicado.
    '''
    
    # Tamanho do patch e número de canais dos patches e número de patches e Quantidade de pixels em um subpatch
    patch_size = x_train.shape[1] # Patch Quadrado
    n_channels = x_train.shape[-1]
    n_patches = x_train.shape[0]
    pixels_patch = patch_size*patch_size
    pixels_subpatch = dim*dim

    if k > pixels_patch//pixels_subpatch:
        raise Exception('k é maior que o número total de subpatches com o tamanho subpatch_size. '
                        'Diminua o tamanho do k ou diminua o tamanho do subpatch')
        
    # Variáveis para extração dos subpatches (tiles) a partir dos patches
    sizes = [1, dim, dim, 1]
    strides = [1, dim, dim, 1]
    rates = [1, 1, 1, 1]
    padding = 'VALID'
    
    # Extrai subpatches e faz o reshape para estar 
    # na forma (patch, número total de subpatches, altura do subpatch, 
    # largura do subpatch, número de canais do subpatch)
    # Na forma como é extraído, o subpatch fica achatado na forma
    # (patch, número vertical de subpatches, número horizontal de subpatches, número de pixels do subpatch achatado)
    subpatches_x_train = tf.image.extract_patches(x_train, sizes=sizes, strides=strides, rates=rates, padding=padding).numpy()
    n_vertical_subpatches = subpatches_x_train.shape[1]
    n_horizontal_subpatches = subpatches_x_train.shape[2]
    subpatches_x_train = subpatches_x_train.reshape((n_patches, n_vertical_subpatches*n_horizontal_subpatches, 
                                                     dim, dim, n_channels))
    
    subpatches_y_train = tf.image.extract_patches(y_train, sizes=sizes, strides=strides, rates=rates, padding=padding).numpy()
    subpatches_y_train = subpatches_y_train.reshape((n_patches, n_vertical_subpatches*n_horizontal_subpatches, 
                                                     dim, dim, 1)) # Só um canal para referência
    

    
    # Serão selecionados, preferencialmente, subpatches com número de pixels de estrada maior que pixels_corte
    pixels_corte = 0

    # Aplica blur nos respectivos subpatches    
    for i_patch in range(len(subpatches_y_train)):
        # Conta número de pixels de estrada em cada subpatch do presente patch
        contagem_estrada = np.array([np.count_nonzero(subpatches_y_train[i_patch, i_subpatch] == 1) 
                                     for i_subpatch in range(subpatches_y_train.shape[1])])
        
        # Array com índices dos subpatches em ordem decrescente de número de pixels de estrada
        indices_sorted_desc = np.argsort(contagem_estrada)[::-1]
        
        # Deixa somente os indices cujos subpatches tem número de pixels maior que o limiar pixels_corte
        indices_maior = np.array([indice for indice in indices_sorted_desc 
                                  if contagem_estrada[indice] > pixels_corte])
        
        # Pega os k subpatches que tem mais pixels de estrada e 
        # que tem quantidade de pixels de estrada maior que 0
        indices_selected_subpatches = indices_maior[:k]
        
        # Converte em lista e coloca em ordem crescente
        indices_selected_subpatches = list(indices_selected_subpatches)
        indices_selected_subpatches.sort()
            
        # Cria array com subpatches escolhidos
        selected_subpatches_x_train_i_patch = subpatches_x_train[i_patch][indices_selected_subpatches]
        
        # Aplica filtro aos subpatches 
        selected_subpatches_x_train_i_patch_blurred = gaussian_filter(selected_subpatches_x_train_i_patch.astype(np.float32), sigma=(0, blur, blur, 0))
        
        # Substitui subpatches pelos respectivos subpatches com blur no array original de subpatches
        subpatches_x_train[i_patch][indices_selected_subpatches] = selected_subpatches_x_train_i_patch_blurred
        
    
    # Agora faz reconstituição dos patches originais, com o devido blur nos subpatches
    # Coloca no formato aceito da função de reconstituição
    x_train_blur = np.zeros(x_train.shape, dtype=np.float16)
    
    for i_patch in range(len(subpatches_x_train)):
        # Pega subpatches do patch
        sub = subpatches_x_train[i_patch]
    
        # Divide em linhas    
        rows = np.split(sub, patch_size//dim, axis=0)
        
        # Concatena linhas
        rows = [tf.concat(tf.unstack(x), axis=1).numpy() for x in rows]
        
        # Reconstroi
        reconstructed = (tf.concat(rows, axis=0)).numpy()
        
        # Atribui
        x_train_blur[i_patch] = reconstructed
    
    # Make predictions
    pred_train_blur, _ = Test_Step(model, x_train_blur)

    # Retorna patches com blur e predições
    return x_train_blur, pred_train_blur
    


def occlude_y_patches(y_train, dim, k):
    # Tamanho do patch e número de canais dos patches e número de patches e Quantidade de pixels em um subpatch
    patch_size = y_train.shape[1] # Patch Quadrado
    n_channels = 1 # Só um canal, pois é máscara
    n_patches = y_train.shape[0]
    pixels_patch = patch_size*patch_size
    pixels_subpatch = dim*dim
    
    if k > pixels_patch//pixels_subpatch:
        raise Exception('k é maior que o número total de subpatches com o tamanho subpatch_size. '
                        'Diminua o tamanho do k ou diminua o tamanho do subpatch')
        
    # Variáveis para extração dos subpatches (tiles) a partir dos patches
    sizes = [1, dim, dim, 1]
    strides = [1, dim, dim, 1]
    rates = [1, 1, 1, 1]
    padding = 'VALID'
    
    
    # Extrai subpatches e faz o reshape para estar 
    # na forma (patch, número total de subpatches, altura do subpatch, 
    # largura do subpatch, número de canais do subpatch)
    # Na forma como é extraído, o subpatch fica achatado na forma
    # (patch, número vertical de subpatches, número horizontal de subpatches, número de pixels do subpatch achatado)
    subpatches_y_train = tf.image.extract_patches(y_train, sizes=sizes, strides=strides, rates=rates, padding=padding).numpy()
    n_vertical_subpatches = subpatches_y_train.shape[1]
    n_horizontal_subpatches = subpatches_y_train.shape[2]
    subpatches_y_train = subpatches_y_train.reshape((n_patches, n_vertical_subpatches*n_horizontal_subpatches, 
                                                     dim, dim, n_channels)) # Só um canal para referência
    
    
    # Serão selecionados, preferencialmente, subpatches com número de pixels de estrada maior que pixels_corte
    pixels_corte = 0
    
    # Aplica blur nos respectivos subpatches    
    for i_patch in range(len(subpatches_y_train)):
        # Conta número de pixels de estrada em cada subpatch do presente patch
        contagem_estrada = np.array([np.count_nonzero(subpatches_y_train[i_patch, i_subpatch] == 1) 
                                     for i_subpatch in range(subpatches_y_train.shape[1])])
        
        # Array com índices dos subpatches em ordem decrescente de número de pixels de estrada
        indices_sorted_desc = np.argsort(contagem_estrada)[::-1]
        
        # Deixa somente os indices cujos subpatches tem número de pixels maior que o limiar pixels_corte
        indices_maior = np.array([indice for indice in indices_sorted_desc 
                                  if contagem_estrada[indice] > pixels_corte])
        # indices_maior = indices_sorted_desc[contagem_estrada > pixels_corte]
        
        # Pega os k subpatches que tem mais pixels de estrada e 
        # que tem quantidade de pixels de estrada maior que 0
        indices_selected_subpatches = indices_maior[:k]
        
        # Converte em lista e coloca em ordem crescente
        indices_selected_subpatches = list(indices_selected_subpatches)
        indices_selected_subpatches.sort()
            
        # Cria array com subpatches escolhidos
        selected_subpatches_y_train_i_patch = subpatches_y_train[i_patch][indices_selected_subpatches]
        
        # Aplica filtro aos subpatches 
        selected_subpatches_y_train_i_patch_occluded = np.zeros(selected_subpatches_y_train_i_patch.shape)
        
        # Substitui subpatches pelos respectivos subpatches com blur no array original de subpatches
        subpatches_y_train[i_patch][indices_selected_subpatches] = selected_subpatches_y_train_i_patch_occluded
        
    
    # Agora faz reconstituição dos patches originais, com o devido blur nos subpatches
    # Coloca no formato aceito da função de reconstituição
    y_train_occluded = np.zeros(y_train.shape, dtype=np.uint8)
    
    for i_patch in range(len(subpatches_y_train)):
        # Pega subpatches do patch
        sub = subpatches_y_train[i_patch]
    
        # Divide em linhas    
        rows = np.split(sub, patch_size//dim, axis=0)
        
        # Concatena linhas
        rows = [tf.concat(tf.unstack(x), axis=1).numpy() for x in rows]
        
        # Reconstroi
        reconstructed = (tf.concat(rows, axis=0)).numpy()
        
        # Atribui
        y_train_occluded[i_patch] = reconstructed
        
    return y_train_occluded
    
    

# intersection between line(p1, p2) and line(p3, p4)
# Extraído de https://gist.github.com/kylemcdonald/6132fc1c29fd3767691442ba4bc84018
def intersect(p1, p2, p3, p4):
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    x4,y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return (x,y)



def plota_curva_precision_recall_relaxada(y, prob, buffer_y, buffer_px=3, num_pontos=10, output_dir='',
                                          save_figure=True):
    '''
    

    Parameters
    ----------
    y : TYPE
        DESCRIPTION. Array de Referência
    prob : TYPE
        DESCRIPTION. Array de Probabilidades gerado a partir do modelo
    buffer_y : TYPE
        DESCRIPTION. Buffer de  buffer_px do Array de Referência 
    buffer_px : TYPE
        DESCRIPTION. Valor do buffer em pixels, exemplo: 1, 2, 3.
    num_pontos : TYPE, optional
        DESCRIPTION. The default is 100. Número de pontos para gerar no gráfico. 
        A ligação desses pontos formará a curva precision-recall relaxada

    Returns
    -------
    precision_scores
        DESCRIPTION. sequência de precisões para cada limiar
    recall_scores
        DESCRIPTION. sequência de recalls para cada limiar
    intersection_found
        DESCRIPTION. ponto em que precision=recall, ou seja, que intercepta a reta em que
        precision=recall, chamado de breakeven point.        


    '''    
    # Achata arrays
    y_flat = np.reshape(y, (y.shape[0] * y.shape[1] * y.shape[2]))
    buffer_y_flat = np.reshape(buffer_y, (buffer_y.shape[0] * buffer_y.shape[1] * buffer_y.shape[2]))
    
    # Define sequências de limiares que serão usados para gerar 
    # resultados de precision e recall
    probability_thresholds = np.linspace(0, 1, num=num_pontos)
    
    # Lista de Precisões e Recalls
    precision_scores = []
    recall_scores = []
    
    # Cria diretório para armazenar os buffers (por questão de performance)
    # Salva e carrega o buffer
    #prob_temp_dir = os.path.join(output_dir, 'prob_temp')
    #os.makedirs(prob_temp_dir, exist_ok=True)
    
    # Percorre probabilidades e calcula precisão e recall para cada uma delas
    # Adiciona cada uma a sua respectiva lista
    for i, p in enumerate(probability_thresholds):
        print(f'Calculando resultados {i+1}/{num_pontos}')
        # Predição e Buffer da Predição
        pred_for_prob = (prob > p).astype('uint8')
        #np.save(os.path.join(prob_temp_dir, f'prob_temp_list{i}.npy'), pred_for_prob)
        #pred_for_prob = np.load(os.path.join(prob_temp_dir, f'prob_temp_list{i}.npy')) 
        buffer_for_prob = buffer_patches(pred_for_prob, dist_cells=buffer_px)
       
        
        # Achatamento da Predição e Buffer da Predição, para entrada na função
        pred_for_prob_flat = np.reshape(pred_for_prob, (pred_for_prob.shape[0] * pred_for_prob.shape[1] * pred_for_prob.shape[2]))
        buffer_for_prob_flat = np.reshape(buffer_for_prob, (buffer_for_prob.shape[0] * buffer_for_prob.shape[1] * buffer_for_prob.shape[2]))
        
        # Cálculo da precisão e recall relaxados
        relaxed_precision = precision_score(buffer_y_flat, pred_for_prob_flat, pos_label=1, zero_division=1)
        relaxed_recall = recall_score(y_flat, buffer_for_prob_flat, pos_label=1, zero_division=1)
        
        # Adiciona precisão e recall às suas listas
        precision_scores.append(relaxed_precision)
        recall_scores.append(relaxed_recall)
        
        
    # Segmento de Linha (p1, p2) pertencerá à reta precision=recall
    # ou seja, está na diagonal do gráfico 
    p1 = (0,0)
    p2 = (1,1)
    
    # Acha algum segmento que faz interseção com a linha em que precision=recall        
    intersections = []
        
    for i in range(len(precision_scores)-1):
        print(i)
        # p3 vai ser o ponto atual da sequência e p4 será o próximo ponto
        p3 = (precision_scores[i], recall_scores[i])
        p4 = (precision_scores[i+1], recall_scores[i+1])
        
        print(p3, p4)
        
        # Adiciona à lists de interseções
        intersection = intersect(p1, p2, p3, p4)
        if intersection:
            intersections.append(intersection)
        else:
            intersections.append(False)
            
    # Pega a primeira interseção da lista caso múltiplas forem achadas        
    intersection_found = [inter for inter in intersections if inter is not False][0]
        
    # Exporta figura
    fig, ax = plt.subplots()
    ax.plot(precision_scores, recall_scores)
    ax.plot(intersection_found[0], intersection_found[1], marker="o", markersize=10)
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.set_title('Curva Precision x Recall')

    if save_figure:    
        fig.savefig(output_dir + f'curva_precision_recall_{buffer_px}px.png')
    
    plt.show()

    # Salva resultados em um dicionário
    curva_precision_recall_results = {'precision_scores':precision_scores,
                                      'recall_scores':recall_scores,
                                      'intersection_found':intersection_found}


    with open(output_dir + 'curva_precision_recall_results' + '.pickle', "wb") as fp: # Salva dicionário com métricas do resultado do experimento
        pickle.dump(curva_precision_recall_results, fp)
        
    # Imprime em arquivo de resultado
    with open(output_dir + f'relaxed_metrics_{buffer_px}px.txt', 'a') as f:
        print('\nBreakeven point in Precision-Recall Curve for Teste', file=f)
        print('=======', file=f)
        print('Precision=Recall in point: (%.4f, %.4f)' % (intersection_found[0], intersection_found[1]), file=f)
        print() 
        
        
    return precision_scores, recall_scores, intersection_found




def treina_com_subpatches(filenames_train_list, filenames_valid_list, filenames_test_list, model_dir):
    # Treina modelos
    for (name_train, name_valid, name_test) in zip(filenames_train_list, filenames_valid_list, filenames_test_list):
        # Move arquivo para diretório de modelos
        shutil.copy(name_train, model_dir + 'x_train.npy')
        shutil.copy(name_valid, model_dir + 'x_valid.npy')
        shutil.copy(name_test, model_dir + 'x_test.npy')
        
        # Treina modelo com os dados
        output_dir = os.path.dirname(name_train)
        treina_modelo(model_dir, output_dir, epochs=1000, early_loss=False, model_type='resunet',
                      loss='cross', lr_decay=True)
        
        
# Etapa 1 se refere ao processamento no qual a entrada do treinamento do pós-processamento é a predição do modelo
# Etapa 2 ao processamento no qual a entrada do treinamento do pós-processamento é a imagem original mais a predição do modelo
# Etapa 3 ao processamento no qual a entrada do treinamento do pós-processamento é o rótulo degradado
# Etapa 5 se refere ao pós-processamento
def gera_dados_segunda_rede(input_dir, y_dir, output_dir, etapa=2):
    x_train = np.load(input_dir + 'x_train.npy')
    x_valid = np.load(input_dir + 'x_valid.npy')
    
    y_train = np.load(y_dir + 'y_train.npy')
    y_valid = np.load(y_dir + 'y_valid.npy')
    
    # Copia y_train e y_valid para diretório de saída, pois eles são mantidos como rótulos da Segunda Rede
    # para o pós-processamento
    # if etapa==1 or etapa==2 or etapa==3 or etapa==4:
    #     shutil.copy(input_dir + 'y_train.npy', output_dir + 'y_train.npy')
    #     shutil.copy(input_dir + 'y_valid.npy', output_dir + 'y_valid.npy')
        
    # Predições
    # if not os.path.exists(os.path.join(output_dir, 'pred_train.npy')):
    #     # Nome do modelo salvo
    #     best_model_filename = 'best_model'
        
    #     # Load model
    #     model = load_model(output_dir + best_model_filename + '.h5', compile=False)        
        
    #     # Faz predição
    #     pred_train, _ = Test_Step(model, x_train, 2)
        
    #     # Converte para tipo que ocupa menos espaço e salva
    #     pred_train = pred_train.astype(np.uint8)
    #     salva_arrays(output_dir, pred_train=pred_train)
    
    if etapa != 3:
        pred_train = np.load(output_dir + 'pred_train.npy')
        pred_valid = np.load(output_dir + 'pred_valid.npy')
        
    # Forma e Copia para Nova Pasta o Novo x_train e x_valid
    if etapa == 1:
        if not os.path.exists(os.path.join(output_dir, 'x_train.npy')) or \
            not os.path.exists(os.path.join(output_dir, 'x_valid.npy')):
            shutil.copy(output_dir + 'pred_train.npy', output_dir + 'x_train.npy')
            shutil.copy(output_dir + 'pred_valid.npy', output_dir + 'x_valid.npy')
    
    if etapa == 2:
        if not os.path.exists(os.path.join(output_dir, 'x_train.npy')) or \
            not os.path.exists(os.path.join(output_dir, 'x_valid.npy')):
            x_train_new = np.concatenate((x_train, pred_train), axis=-1)
            x_valid_new = np.concatenate((x_valid, pred_valid), axis=-1)
            salva_arrays(output_dir, x_train=x_train_new, x_valid=x_valid_new)
            
    if etapa == 3:
        dim = 14
        k = 10
        if not os.path.exists(os.path.join(output_dir, 'x_train.npy')) or \
            not os.path.exists(os.path.join(output_dir, 'x_valid.npy')):
            x_train_new = occlude_y_patches(y_train, dim, k)
            x_valid_new = occlude_y_patches(y_valid, dim, k)
            salva_arrays(output_dir, x_train=x_train_new, x_valid=x_valid_new)
            
    # Libera memória
    if all(pred in locals() for pred in ('pred_train', 'pred_valid')):
        del x_train, x_valid, y_train, y_valid, pred_train, pred_valid
    else:
        del x_train, x_valid, y_train, y_valid
        
    gc.collect()
            
            
    # Lê arrays referentes aos patches de teste
    x_test = np.load(input_dir + 'x_test.npy')
    
    
    # Copia y_test para diretório de saída, pois ele será usado como entrada (rótulo) para o pós-processamento
    # if etapa==1 or etapa==2 or etapa==3 or etapa==4:
    #     shutil.copy(input_dir + 'y_test.npy', output_dir + 'y_test.npy')
        
    # Predições
    pred_test = np.load(output_dir + 'pred_test.npy')
        
    # Forma e Copia para Nova Pasta o Novo x_test
    if etapa == 1:
        if not os.path.exists(os.path.join(output_dir, 'x_test.npy')):
            shutil.copy(output_dir + 'pred_test.npy', output_dir + 'x_test.npy')

    if etapa == 2:
        if not os.path.exists(os.path.join(output_dir, 'x_test.npy')):
            x_test_new = np.concatenate((x_test, pred_test), axis=-1)
            salva_arrays(output_dir, x_test=x_test_new)
            
    if etapa == 3: # No caso o X teste será igual ao da etapa 1
        if not os.path.exists(os.path.join(output_dir, 'x_test.npy')):
            shutil.copy(output_dir + 'pred_test.npy', output_dir + 'x_test.npy')
            
    # Libera memória
    del x_test, pred_test
    gc.collect()
    
    
    
def stack_uneven(arrays, fill_value=0.):
    '''
    Fits arrays into a single numpy array, even if they are
    different sizes. `fill_value` is the default value.

    Args:
            arrays: list of np arrays of various sizes
                (must be same rank, but not necessarily same size)
            fill_value (float, optional):

    Returns:
            np.ndarray
    '''
    sizes = [a.shape for a in arrays]
    max_sizes = np.max(list(zip(*sizes)), -1)
    # The resultant array has stacked on the first dimension
    result = np.full((len(arrays),) + tuple(max_sizes), fill_value)
    for i, a in enumerate(arrays):
      # The shape of this array `a`, turned into slices
      slices = tuple(slice(0,s) for s in sizes[i])
      # Overwrite a block slice of `result` with this array `a`
      result[i][slices] = a
    return result


def extract_difference_reftiles(tile_before: str, tile_after: str, out_raster_path:str, buffer_px: int=3):
    '''
    Extrai diferença entre tiles de referência de diferentes épocas. 
    Ela é computada assim: Tile Depois - Buffer(Tile Antes).
    
    Exemplo: extract_difference_reftiles('tiles\\masks\\2016\\test\\reftile_2016_15.tif', 
                                         'tiles\\masks\\2018\\test\\reftile_2018_15.tif',
                                         'tiles\\masks\\Diff\\test\\diff_reftile_2018_2016_15.tif')

    Parameters
    ----------
    tile_before : str
        DESCRIPTION.
    tile_after : str
        DESCRIPTION.
    out_raster_path : str
        DESCRIPTION.
    buffer_px : int, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    None, but save the tile difference as raster .tif in the out_raster_path.

    '''
    # Open Images with GDAL
    gdal_header_tile_before = gdal.Open(str(tile_before))
    gdal_header_tile_after = gdal.Open(str(tile_after)) 
    
    # X and Y size of image
    xsize = gdal_header_tile_before.RasterXSize
    ysize = gdal_header_tile_after.RasterYSize
    
    # Read as Rasters as Numpy Array
    tile_before_arr = gdal_header_tile_before.ReadAsArray()
    tile_after_arr = gdal_header_tile_after.ReadAsArray()
    
    # Replace NODATA with 0s to make calculations
    # Replace only Array After because the Buffer of Array Before already eliminates Nodata
    tile_after_arr_0 = tile_after_arr.copy()
    tile_after_arr_0[tile_after_arr_0==255] = 0
    
    # Make Buffer on Image Before and Subtract it from Image After
    # Calculate Difference: Tile After - Buffer(Tile Before)
    # Consider only Positive Values (New Roads)
    dist_buffer_px = 3
    buffer_before = array_buffer(tile_before_arr, dist_buffer_px)
    diff_after_before = np.clip(tile_after_arr_0.astype(np.int16) - buffer_before.astype(np.int16), 0, 1)
    diff_after_before = diff_after_before.astype(np.uint8)
    diff_after_before[tile_after_arr==255] = 0 # Set Original Nodata values to 0, output doesn't contain Nodata
    
    # Export Raster
    driver = gdal.GetDriverByName('GTiff') # Geotiff Driver
    file = driver.Create(out_raster_path, xsize, ysize, bands=1, eType=gdal.GDT_Byte) # Tipo Unsigned Int 8 bits
    
    file_band = file.GetRasterBand(1) 
    file_band.WriteArray(diff_after_before)
    
    file.SetGeoTransform(gdal_header_tile_after.GetGeoTransform())
    file.SetProjection(gdal_header_tile_after.GetProjection())    
    
    file.FlushCache()