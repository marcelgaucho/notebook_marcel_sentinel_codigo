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

from tensorflow.keras.layers import Input, concatenate, Conv2D, BatchNormalization, Activation 
from tensorflow.keras.layers import MaxPool2D, Conv2DTranspose, UpSampling2D, Concatenate
from tensorflow.keras.layers import Dropout, Add
from tensorflow.keras.models import Model, load_model

from sklearn.utils import shuffle

from sklearn.metrics import (confusion_matrix, f1_score, precision_score, 
                            recall_score, accuracy_score, ConfusionMatrixDisplay)

from scipy.ndimage import gaussian_filter

from tensorflow.keras.optimizers import Adam
from focal_loss import SparseCategoricalFocalLoss



import matplotlib.pyplot as plt

def load_tiff_image(image):
    print(image)
    gdal_header = gdal.Open(image)
    img_gdal = gdal_header.ReadAsArray()
    img = np.transpose(img_gdal, (1,2,0)) # Transpõe imagem para as bandas ficarem
                                          # na última dimensão
    print(img.shape)
    return img


def load_tiff_image_reference(image):
    print (image)
    gdal_header = gdal.Open(image)
    img_gdal = gdal_header.ReadAsArray()
    #img = np.expand_dims(img_gdal, 2)
    img = img_gdal
    print(img.shape)
    return img


def normalization(image):
    # Primeiro remocela a imagem planificando as linhas e colunas para fazer normalização
    # Depois remodela para formato original
    image_reshaped = image.reshape(image.shape[0]*image.shape[1], image.shape[2])
    scaler = MinMaxScaler(feature_range=(0,1))
    image_normalized_ = scaler.fit_transform(image_reshaped)
    image_normalized = image_normalized_.reshape(image.shape[0], image.shape[1], image.shape[2])
    
    return image_normalized


# Função para extrair os patches
def extract_patches(image, reference, patch_size, stride, border_patches=False):
    '''
    Function: extract_patches
    -------------------------
    Extract patches from the original and reference image
    
    Input parameters:
      image      = array containing the original image (h,w,c)
      reference  = array containing the reference image (h,w)
      patch_size = patch size (scalar). The shape of the patch is square.
      stride     = displacement to be applied.
      border_patches = include patches overlaping image borders (at most only one for each line or column and only when necessary to complete 
                                                                 the image)
    
    Returns: 
      A, B = List containing the patches for the input image (A) and respective reference (B).
    '''
    # Listas com os patches da imagem e da referência
    patch_img = []
    patch_ref = []
    
    # Quantidade de canais da imagem (canais na última dimensão - channels last)
    image_channels = image.shape[-1]
    
    # Altura e Largura percorrendo a imagem com o stride
    h = math.ceil(image.shape[0] / stride)
    w = math.ceil(reference.shape[1] / stride)
    
    
    # Acha primeiro valor de m e n que completa ou transborda a imagem 
    # Imagem terá math.floor(image.shape[0] / stride) patches na vertical e 
    # math.floor(image.shape[1] / stride) patches na horizontal, caso border_pathes=False.
    # Imagem terá firstm_out+1 patches na vertical e 
    # firstn_out+1 patches na horizontal, caso border_pathes=True.
    for m in range(0, h):
        i_h = m*stride
        if i_h + patch_size == image.shape[0]:
            break
        if i_h + patch_size > image.shape[0]: 
            break
    
    firstm_out = m
    
    for n in range(0, w):
        i_w = n*stride
        if i_w + patch_size == image.shape[1]: 
            break
        if i_w + patch_size > image.shape[1]: 
            break
    
    firstn_out = n
    
    
    # Percorre dimensões obtidas com stride
    for m in range(0, h):
        for n in range(0, w):
            # Índices relativos à altura e largura percorrendo com stride
            i_h = m*stride
            i_w = n*stride
            
            # Adiciona Patch da Imagem e Referência caso ele esteja contido na imagem de entrada
            #print('M %d, N %d, Height start %d finish %d , Width start %d finish %d' % (m, n, i_h , i_h+patch_size, i_w, i_w+patch_size) )
            if ( (i_h + patch_size <= image.shape[0]) and (i_w + patch_size <= image.shape[1]) ):
                patch_img.append( image[i_h : i_h+patch_size, i_w : i_w+patch_size, :] )
                patch_ref.append( reference[i_h : i_h+patch_size, i_w : i_w+patch_size] )
                
            # Trata bordas no caso que o patch esteja parcialmente contido na imagem de entrada (parte do patch está fora da borda)
            # Preenche o que sai da borda com 0s
            elif border_patches:
                # Inicia Patches de Borda (imagem e referência) com 0s 
                border_patch_img = np.zeros((patch_size, patch_size, image_channels))
                border_patch_ref = np.zeros((patch_size, patch_size))
                
                # Se patch ultrapassa altura da imagem,
                # border_mmax é o que falta desde o início do patch até a borda inferior da imagem
                if (i_h + patch_size > image.shape[0]):
                    border_mmax = image.shape[0] - i_h
                # Caso contrário mantém-se o tamanho do patch   
                else:
                    border_mmax = patch_size
                
                # Se patch ultrapassa largura da imagem,
                # border_nmax é o que falta desde o início do patch até a borda direita da imagem     
                if (i_w + patch_size > image.shape[1]):
                    border_nmax = image.shape[1] - i_w
                else:
                    border_nmax = patch_size
                    
                    
                # Preenche patches
                border_patch_img[0:border_mmax, 0:border_nmax, :] = image[i_h : i_h+border_mmax, i_w : i_w+border_nmax, :]
                border_patch_ref[0:border_mmax, 0:border_nmax] = reference[i_h : i_h+border_mmax, i_w : i_w+border_nmax]
                
                
                # Adiciona patches à lista somente se m e n forem os menores valores a transbordar a imagem
                if m <= firstm_out and n <= firstn_out:
                    patch_img.append( border_patch_img )
                    patch_ref.append( border_patch_ref )
      
        
    # Retorna os arrays de patches da imagem e da referência        
    return np.array(patch_img), np.array(patch_ref)

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



def resnet_block_chamorro(x, n_filter, ind):
    x_init = x

    ## Conv 1
    x = Conv2D(n_filter, (3, 3), activation='relu', padding="same", name = 'res1_net'+str(ind))(x)
    x = Dropout(0.5, name = 'drop_net'+str(ind))(x)
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
    
    else:
        raise Exception("Model options are 'unet' and 'resunet' and 'resunet chamorro' and 'corred' ")
        




    


# Define x_datagen e y_datagen, se for usar aumento de dados (aqui não está usando)
x_datagen = None
y_datagen = None


def set_number_of_batches(qt_train_samples, qt_valid_samples, batch_size, data_augmentation, number_samples_for_generator=6):
    # In case of data augmentation, the number of batches
    # will be decided by the number of samples for the 
    # augmentation generator  
    if data_augmentation:
        train_batchs_qtd = qt_train_samples//number_samples_for_generator
        valid_batchs_qtd = qt_valid_samples//number_samples_for_generator
    # Else it is decided by the batch size
    else:
        train_batchs_qtd = qt_train_samples//batch_size
        valid_batchs_qtd = qt_valid_samples//batch_size

    return train_batchs_qtd, valid_batchs_qtd


def get_batch_samples(x, y, batch, batch_size, data_augmentation, number_samples_for_generator):
    if data_augmentation:
        x_batch = x[batch * number_samples_for_generator : (batch + 1) * number_samples_for_generator, : , : , :]
        y_batch = y[batch * number_samples_for_generator : (batch + 1) * number_samples_for_generator, : , : , :]

        x_iterator = x_datagen.flow(x_batch, seed=batch)
        y_iterator = y_datagen.flow(y_batch, seed=batch)

        x_batch = np.array([next(x_iterator)[0] for _ in range(batch_size)])
        y_batch = np.array([next(y_iterator)[0] for _ in range(batch_size)])
    else:
        x_batch = x[batch * batch_size : (batch + 1) * batch_size, : , : , :]
        y_batch = y[batch * batch_size : (batch + 1) * batch_size, : , : , :]

    return x_batch, y_batch



def train_unet(net, x_train, y_train, x_valid, y_valid, batch_size, epochs, early_stopping_epochs, early_stopping_delta, filepath, filename, data_augmentation=False, number_samples_for_generator=1, early_stopping=True, early_loss=True):
    print('Start the training...')

    # calculating number of batches
    train_batchs_qtd, valid_batchs_qtd = set_number_of_batches(x_train.shape[0], x_valid.shape[0], batch_size, data_augmentation, number_samples_for_generator)
  
    history_train = [] # Loss and measure of quality (accuracy, recall, etc) for each epoch will be stored on the list
    history_valid = []
    valid_loss_best_model = float('inf')
    valid_metric_best_model = 1e-20
    no_improvement_count = 0

    for epoch in range(epochs):
        print('Start epoch ... %d ' %(epoch) )
        # shuffle train set in each epoch
        #x_train, y_train = shuffle(x_train , y_train, random_state = 0)
        x_train, y_train = shuffle(x_train , y_train)

        # TRAINING
        train_loss = np.zeros((1 , 2)) # Initialize array that will be used to store the pair (loss, measure of quality)
        # mini batches strategy (iterate through batches)
        for  batch in range(train_batchs_qtd):
            print('Start batch ... %d ' %(batch) )
            x_train_batch, y_train_batch = get_batch_samples(x_train, y_train, batch, batch_size, data_augmentation, number_samples_for_generator) # Pega slice do array referente ao batch sendo treinado
            train_loss = train_loss + net.train_on_batch(x_train_batch, y_train_batch) # Accumulate loss and measure of quality
            

        # Estimating the loss and measure of quality in the training set
        # Divide total loss and measure of quality by the number of batches (average)
        train_loss = train_loss/train_batchs_qtd

        # VALIDATING
        # shuffle valid set in each epoch
        #x_valid, y_valid = shuffle(x_valid , y_valid, random_state = 0)
        x_valid, y_valid = shuffle(x_valid , y_valid)
        
        valid_loss = np.zeros((1 , 2))
        # Evaluating the network (model) with the validation set
        for  batch in range(valid_batchs_qtd):
            x_valid_batch, y_valid_batch = get_batch_samples(x_valid, y_valid, batch, batch_size, data_augmentation, number_samples_for_generator)
            valid_loss = valid_loss + net.test_on_batch(x_valid_batch, y_valid_batch)
            

        # Estimating the loss in the validation set
        valid_loss = valid_loss/valid_batchs_qtd

        # Showing the results (Loss and accuracy)
        print("%d [training loss: %f , Train %s: %.2f%%][Test loss: %f , Test %s:%.2f%%]" %(epoch , train_loss[0 , 0], net.metrics_names[1], 100*train_loss[0 , 1] , 
                                                                                              valid_loss[0 , 0] , net.metrics_names[1], 100 * valid_loss[0 , 1]))
        
        history_train.append( train_loss )
        history_valid.append( valid_loss )

        print(history_train)
        print(history_valid)

        # Early Stopping
        # valid_loss[0 , 0]/valid_loss_best_model  
        # is the fraction of loss of the current epoch compared with 
        # the last best epoch.
        # If this number is near to 1 or even larger than 1,
        # then 1 minus this value will be near to 0
        # or even negative. In resume, it will be less
        # than early_stopping_delta
        
        if early_stopping:
            # Early Stopping on Loss 
            if early_loss:
                # Se valor absoluto da fração de incremento (ou decremento) for abaixo de early_stopping_delta,
                # então segue para contagem do Early Stopping
                fraction = 1-(valid_loss[0 , 0]/valid_loss_best_model)
                if abs(fraction) < early_stopping_delta:
                    # Stop if there are no improvement along early_stopping_epochs  
                    # This means, the situation above described persists for
                    # early_stopping_epochs
                    print('Early Stopping Count Increasing to: %d' % (no_improvement_count+1))
                    if no_improvement_count+1 >= early_stopping_epochs:
                        print('Early Stopping reached')
                        break
                    else:
                        no_improvement_count = no_improvement_count+1
                        
                # Perda aumentando, pois loss/loss_best_model é maior que 1
                # Também segue para contagem do Early Stopping
                elif fraction < 0:
                    # Stop if there are no improvement along early_stopping_epochs  
                    # This means, the situation above described persists for
                    # early_stopping_epochs
                    print('Early Stopping Count Increasing to: %d' % (no_improvement_count+1))
                    if no_improvement_count+1 >= early_stopping_epochs:
                        print('Early Stopping reached')
                        break
                    else:
                        no_improvement_count = no_improvement_count+1
    
                # Perda diminuindo, pois loss/loss_best_model é menor que 1
                # Nesse caso salva o modelo e atualiza o valor de perda do melhor modelo
                else:
                    valid_loss_best_model = valid_loss[0 , 0]
                    no_improvement_count = 0
        
                    # Saving best model  
                    print("Saving the model...")
                    net.save(filepath+filename+'.h5')
                    
            # Similar thing on metric
            else:
                # Se valor absoluto da fração de incremento (ou decremento) for abaixo de early_stopping_delta,
                # então segue para contagem do Early Stopping
                # Aqui, em fraction, a diferença é ao contrário, pois queremos que uma fração positiva aumente a métrica (o que é considerado bom)
                fraction = (valid_loss[0 , 1]/valid_metric_best_model)-1
                if abs(fraction) < early_stopping_delta:
                    # Stop if there are no improvement along early_stopping_epochs  
                    # This means, the situation above described persists for
                    # early_stopping_epochs
                    print('Early Stopping Count Increasing to: %d' % (no_improvement_count+1))
                    if no_improvement_count+1 >= early_stopping_epochs:
                        print('Early Stopping reached')
                        break
                    else:
                        no_improvement_count = no_improvement_count+1
                        
                # Métrica diminuindo, pois loss/loss_best_model é menor que 1
                # Como isso normalmente é ruim, também segue para contagem do Early Stopping
                elif fraction < 0:
                    # Stop if there are no improvement along early_stopping_epochs  
                    # This means, the situation above described persists for
                    # early_stopping_epochs
                    print('Early Stopping Count Increasing to: %d' % (no_improvement_count+1))
                    if no_improvement_count+1 >= early_stopping_epochs:
                        print('Early Stopping reached')
                        break
                    else:
                        no_improvement_count = no_improvement_count+1
                        
                # Métrica aumentando, pois loss/loss_best_model é maior que 1
                # Normalmente é uma coisa positiva, como por exemplo acurácia, precisão, recall aumentando.
                # Então salvamos o modelo
                else:
                    valid_loss_best_model = valid_loss[0 , 1]
                    no_improvement_count = 0
        
                    # Saving best model  
                    print("Saving the model...")
                    net.save(filepath+filename+'.h5')
                    
        else:
            # Saving best model  
            print("Saving the model...")
            net.save(filepath+filename+'.h5')
                
    # Return a list of 2 lists
    # The first is the list with the train loss and accuracy
    # The second is the list with the validation loss and accuracy
    return [ history_train, history_valid ]



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

    plt.show()
    

    
    

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
    for i in range(len(patch_test)):
        print('Buffering patch {}/{}'.format(i+1, len(patch_test))) 
        # Patch being buffered 
        patch_batch = patch_test[i, ..., 0]
        
        # Concatenate result patches to form result
        if i == 0:
            result = array_buffer(patch_batch, dist_cells=dist_cells)[np.newaxis, ..., np.newaxis]
        else:
            patch_batch_r_new = array_buffer(patch_batch, dist_cells=dist_cells)[np.newaxis, ..., np.newaxis]
            result = np.concatenate((result, patch_batch_r_new), axis=0)
            
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
def Test_Step(model, patch_test, step, out_sigmoid=False, threshold_sigmoid=0.5):
    n = len(patch_test)
    i = 0
    while i < n:
        print('Predicting batch {:.0f}/{}'.format(i/step+1, n//step + (n%step>0) )) # Round down and up, respectively
        # Patch being transformed by NN 
        patch_batch = patch_test[i:i+step]
        
        # Concatenate result patches to form result
        if i < step:
            result = model.predict(patch_batch)
        else:
            patch_batch_r_new = model.predict(patch_batch)
            result = np.concatenate((result, patch_batch_r_new), axis=0)
        
        i = i+step

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



# Função que faz a extração de tiles em uma imagem
def extract_tiles(input_img, input_ref, xoff, yoff, xsize, ysize, folder):
    # Open image 
    ds = gdal.Open(input_img)
    img_np = ds.ReadAsArray()
    img_np = np.transpose(img_np, (1,2,0))
    img_np = normalization(img_np)
    ds_ref = gdal.Open(input_ref)
    img_ref_np = ds_ref.ReadAsArray()
    
    # GeoTransform data
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = gdal_array.OpenArray(np.transpose(img_np, (2,0,1)).astype(np.float32)) # Update dataset to normalized image
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    
    print('proj img ', proj)
    print('Geotransform img ', gt)
    

    # Coordinates of upper left corner
    xmin = gt[0]
    ymax = gt[3]
    res = gt[1]
    
    xmin = xmin + res*xoff # Coordinates updated for offset
    ymax = ymax - res*yoff
    
    # Image from (xoff, yoff) to end of image (lower right)
    subimg_np = img_np[xoff:, yoff:, :]
    subimg_ref_np = img_ref_np[xoff:, yoff:]
    
    # Listas com os patches da imagem e da referência
    patch_img = []
    patch_ref = [] 
    
    # Altura e Largura percorrendo a imagem com o stride
    # No caso, onde não há sobreposição, o stride horizontal será igual 
    # a largura da imagem o stride vertical será igual a altura da imagem
    hstride = xsize
    vstride = ysize
    h = math.ceil(img_np.shape[0] / vstride)
    w = math.ceil(img_np.shape[1] / hstride)
    

    # Percorre dimensões obtidas com stride
    for m in range(0, h):
        for n in range(0, w):
            print('Processing tile ', m, n)
            # Índices relativos à altura e largura percorrendo com stride
            i_h = m*vstride
            i_w = n*hstride
            
            # Número de colunas e linhas do resultado (caso o patch não seja adicionado)
            if n == w-1: cols = w-1
            if m == h-1: lines = h-1
            
            # Adiciona Patch da Imagem e Referência caso ele esteja contido na imagem de entrada
            #print('M %d, N %d, Height start %d finish %d , Width start %d finish %d' % (m, n, i_h , i_h+patch_size, i_w, i_w+patch_size) )
            if ( (i_h + ysize <= img_np.shape[0]) and (i_w + xsize <= img_np.shape[1]) ):
                patch_img.append( subimg_np[i_h : i_h+ysize, i_w : i_w+xsize, :] )
                patch_ref.append( subimg_ref_np[i_h : i_h+ysize, i_w : i_w+xsize] )
                
                # Altera número de colunas e linhas do resultado (caso o patch seja adicionado)
                if n == w-1: cols = w
                if m == h-1: lines = h
                
                # Limites x e y do patch
                xmin_tile = xmin + i_w*res 
                xmax_tile = xmin_tile + hstride*res 
                ymax_tile = ymax - i_h*res
                ymin_tile = ymax_tile - vstride*res
                
                print(f'Limites do Patch xmin={xmin_tile}, xmax_tile={xmax_tile}, ymax_tile={ymax_tile}, ymin_tile={ymin_tile}')
                
                # Exporta tiles para pasta
                # Formato linha_coluna para tile (imagem) ou reftile (referência)
                
                tile_name = 'tile_' + str(m) + '_' + str(n) + '.tif'
                tile_ref_name = 'reftile_' + str(m) + '_' + str(n) + '.tif'
                
                gdal.Translate(os.path.join(folder, tile_name), ds, projWin = (xmin_tile, ymax_tile, xmax_tile, ymin_tile), 
                               xRes = res, yRes = -res)
                gdal.Translate(os.path.join(folder, tile_ref_name), ds_ref, projWin = (xmin_tile, ymax_tile, xmax_tile, ymin_tile), 
                               xRes = res, yRes = -res, noData=255)  
                
                
        
    # Retorna os arrays de patches da imagem e da referência        
    return np.array(patch_img), np.array(patch_ref), lines, cols


# Função que retorna os tiles igual ou acima de um certo percentual de pixels de estrada
def filtra_tiles_estradas(img_tiles, ref_tiles, perc_corte):
    '''
    

    Parameters
    ----------
    ref_tiles : TYPE
        DESCRIPTION.
    perc_corte : TYPE
        DESCRIPTION.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    if perc_corte < 0 or perc_corte > 100:
        raise Exception('Percentual de pixels de estrada deve estar entre 0 e 100')
        
    
    # Quantidade de pixels em um tile
    pixels_tile = ref_tiles.shape[1]*ref_tiles.shape[2]

    # Quantidade mínima de pixels de estrada     
    pixels_corte = pixels_tile*(perc_corte/100)
    
    # Esse array mostra a contagem de pixels de estrada para cada tile
    # O objetivo é pegar os tiles que tenham número de pixels de estrada maior que o limiar
    contagem_de_estrada = np.array([np.count_nonzero(ref_tiles[i] == 1) for i in range(ref_tiles.shape[0])])
    
    # Esse array são os índices dos tiles que tem número de pixels de estrada maior que o limiar
    indices_maior = np.where(contagem_de_estrada >= pixels_corte)[0]  
    
    # Filtra arrays
    img_tiles_filt = img_tiles[indices_maior]
    ref_tiles_filt = ref_tiles[indices_maior]
    
    # Retorna arrays filtrados e índices dos arrays escolhidos
    return img_tiles_filt, ref_tiles_filt, indices_maior


# Função que copia os tiles filtrados (por filtra_tiles_estradas) desde uma pasta para outra
# cols é o número de tiles na horizontal, indices_filtro pode ser a variável indices_maior retornada por
# por filtra_tiles_estradas
def copia_tiles_filtrados(tiles_folder, new_tiles_folder, indices_filtro, cols):
    # Acha linha e coluna do respectivo tile
    linhas = []
    colunas = []
    
    for ind in indices_filtro:
        linha = ind // cols
        coluna = ind % cols
        
        linhas.append(linha)
        colunas.append(coluna)
    
    # Lista com os nomes de arquivos dos tiles na pasta de entrada
    tile_files = [f for f in os.listdir(tiles_folder) if os.path.isfile(os.path.join(tiles_folder, f))]
    
    # Copia arquivos
    for tile_file in tile_files:
        filename_wo_ext = tile_file.split('.tif')[0] 
        tile_line = int(filename_wo_ext.split('_')[1])
        tile_col = int(filename_wo_ext.split('_')[2])
        
        for i in range(len(linhas)):
            if tile_line == linhas[i] and tile_col == colunas[i]:
                shutil.copy(os.path.join(tiles_folder, tile_file), new_tiles_folder)
                
            continue
        
# A partir de dicionário com endereço de tile:máscara, é feita extração dos patches no tile        
def extract_patches_from_tiles(imgs_labels_dict, patch_size, patch_stride, border_patches=False):
    tile_items = list(imgs_labels_dict.items())
    
    for i in range(len(tile_items)):
        tile_item = tile_items[i]
        
        img_tile = load_tiff_image(tile_item[0]) 
        #img_tile = normalization(img_tile)
        label_tile = load_tiff_image_reference(tile_item[1])
        
        print(f'tile_items img {i} =' , tile_item[0])
        print(f'tile_items label {i} =' , tile_item[1])
        
        # Concatenate result patches to form result    
        if i == 0:
            x_patches, y_patches = extract_patches(img_tile, label_tile, patch_size, patch_stride, border_patches=border_patches)
        else:
            x_patches_new, y_patches_new = extract_patches(img_tile, label_tile, patch_size, patch_stride, border_patches=border_patches)
            x_patches = np.concatenate((x_patches, x_patches_new), axis=0)
            y_patches = np.concatenate((y_patches, y_patches_new), axis=0)
            

    # Transform y_patches, shape (N, H, W) into shape (N, H, W, C). Necessary for data agumentation.
    y_patches = np.expand_dims(y_patches, 3)
            
    return x_patches, y_patches



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


# Função para treinar o modelo conforme os dados (arrays numpy) em uma pasta de entrada, salvando o modelo e o 
# histórico na pasta de saída
def treina_modelo(input_dir: str, output_dir: str, model_type: str ='resunet', epochs=150, early_stopping=True, 
                  early_loss=True, loss='focal', gamma=2, metric=my_f1score, best_model_filename = 'best_model'):
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

    Returns
    -------
    None.

    '''
    
    # Lê arrays salvos
    '''
    if correction:
        x_train = np.load(input_dir + 'pred_train.npy')
        x_valid = np.load(input_dir + 'pred_valid.npy')
    else:
        x_train = np.load(input_dir + 'x_train.npy')
        x_valid = np.load(input_dir + 'x_valid.npy')
    '''
        
    x_train = np.load(input_dir + 'x_train.npy')
    x_valid = np.load(input_dir + 'x_valid.npy')
    y_train = np.load(input_dir + 'y_train.npy')
    y_valid = np.load(input_dir + 'y_valid.npy')
    
    print('Shape dos arrays:')
    print('Shape x_train: ', x_train.shape)
    print('Shape y_train: ', y_train.shape)
    print('Shape x_valid: ', x_valid.shape)
    print('Shape y_valid: ', y_valid.shape)
    print('')
    
    # Constroi modelo
    # input_shape = (patch_size, patch_size, image_channels)
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    num_classes = 2
    model = build_model(input_shape, num_classes, model_type=model_type)
    # model.summary()
    print('Input Patch Shape = ', input_shape)
    print()
    

    # Definição dos hiperparâmetros
    batch_size = 8
    epochs = epochs

    # Parâmetros do Early Stopping
    early_stopping = early_stopping
    early_stopping_epochs = 50
    early_stopping_delta = 0.0001 # aumento delta (percentual de diminuição da perda) equivalente a 0.01%
    
    # Sem Data Augmentation (Ele é feito direto no conjunto de treinamento)
    data_augmentation = False
    
    # Otimizador
    adam = Adam(learning_rate = 0.0001 , beta_1=0.9)
    
    # Compila o modelo
    if loss == 'focal':
        model.compile(loss = SparseCategoricalFocalLoss(gamma=gamma), optimizer=adam , metrics=[metric])
    elif loss == 'cross':
        model.compile(loss = 'sparse_categorical_crossentropy', optimizer=adam, metrics=[metric])

    print('Hiperparâmetros:')
    print('Modelo:', model_type)
    print('Batch Size:', batch_size)
    print('Epochs:', epochs)
    print('Early Stopping:', early_stopping)
    print('Early Stopping Epochs:', early_stopping_epochs)
    print('Early Stopping Delta:', early_stopping_delta)
    print('Otimizador:', 'Adam')
    print('Learning Rate:', 0.0001)
    print('Beta 1:', 0.9)
    print('Função de Perda:', loss)
    print('Gamma para Focal Loss:', gamma)
    print()
        


    # Nome do modelo a ser salvo
    best_model_filename = best_model_filename
    
    # Treina o modelo
    history = train_unet(model, x_train, y_train, x_valid, y_valid, batch_size, epochs, early_stopping_epochs, early_stopping_delta, 
                         output_dir, best_model_filename, data_augmentation, early_stopping=early_stopping, early_loss=early_loss)

    # Imprime e salva história do treinamento
    print('history = \n', history)
    
    with open(output_dir + 'history_' + best_model_filename + '.txt', 'w') as f:
        f.write('history = \n')
        f.write(str(history))
        
    with open(output_dir + 'history_pickle_' +  best_model_filename + '.pickle', "wb") as fp: # Salva histórico (lista Python) para recuperar depois
        pickle.dump(history, fp)
        




def treina_modelo_ensemble(input_dir: str, output_dir_ensemble: str, n_members: int = 10, model_type: str ='resunet', epochs=150, 
                           early_stopping=True, early_loss=True, loss='focal', gamma=2, metric=my_f1score, 
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
# Retorna as métricas de precisão, recall e f1-score relaxados
# Além disso constroi os mosaicos de resultado 
# Etapa 1 se refere ao processamento no qual a entrada do pós-processamento é a predição do modelo
# Etapa 2 ao processamento no qual a entrada do pós-processamento é a imagem original mais a predição do modelo
# Etapa 3 ao processamento no qual a entrada do pós-processamento é a imagem original com blur gaussiano mais a predição desses dados ruidosos com o modelo
# Etapa 4 se refere ao pós-processamento   
def avalia_modelo(input_dir: str, output_dir: str, metric_name = 'F1-Score', 
                  etapa=3, dist_buffers = [3], std_blur = 0.4):
    metric_name = metric_name
    etapa = etapa
    dist_buffers = dist_buffers
    std_blur = std_blur
    
    x_train = np.load(input_dir + 'x_train.npy')
    y_train = np.load(input_dir + 'y_train.npy')
    x_valid = np.load(input_dir + 'x_valid.npy')
    y_valid = np.load(input_dir + 'y_valid.npy')
    
    # Copia y_train e y_valid para diretório de saída, pois eles serão usados depois como entrada (rótulos)
    # para o pós-processamento
    if etapa==1 or etapa==2 or etapa==3:
        shutil.copy(input_dir + 'y_train.npy', output_dir + 'y_train.npy')
        shutil.copy(input_dir + 'y_valid.npy', output_dir + 'y_valid.npy')
        
        
    # Nome do modelo salvo
    best_model_filename = 'best_model'
    
    # Lê histórico
    with open(output_dir + 'history_pickle_' + best_model_filename + '.pickle', "rb") as fp:   
        history = pickle.load(fp)
    
    # Show and save history
    show_graph_loss_accuracy(np.asarray(history), 1, metric_name = metric_name, save=True, save_path=output_dir)
    
    # Load model
    model = load_model(output_dir + best_model_filename + '.h5', compile=False)
    
    
    # Test the model over training and validation data (outputs result if at least one of the files does not exist)
    if not os.path.exists(os.path.join(output_dir, 'pred_train.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'prob_train.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'pred_valid.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'prob_valid.npy')):
        pred_train, prob_train = Test_Step(model, x_train, 2)
        pred_valid, prob_valid = Test_Step(model, x_valid, 2)
        
        # Probabilidade apenas da classe 1 (de estradas)
        prob_train = prob_train[..., 1:2] 
        
        
        
        prob_valid = prob_valid[..., 1:2]
        
        # Converte para tipos que ocupam menos espaço
        pred_train = pred_train.astype(np.uint8)
        prob_train = prob_train.astype(np.float16)
        pred_valid = pred_valid.astype(np.uint8)
        prob_valid = prob_valid.astype(np.float16)
        
        # Salva arrays de predição do Treinamento e Validação    
        salva_arrays(output_dir, pred_train=pred_train, prob_train=prob_train, 
                     pred_valid=pred_valid, prob_valid=prob_valid)
    
    # Lê arrays de predição do Treinamento e Validação (para o caso deles já terem sido gerados)  
    pred_train = np.load(output_dir + 'pred_train.npy')
    prob_train = np.load(output_dir + 'prob_train.npy')
    pred_valid = np.load(output_dir + 'pred_valid.npy')
    prob_valid = np.load(output_dir + 'prob_valid.npy')
    
    
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
            pred_train_blur, _ = Test_Step(model, x_train_blur, 2)
            pred_valid_blur, _ = Test_Step(model, x_valid_blur, 2)
            x_train_new = np.concatenate((x_train_blur, pred_train_blur), axis=-1)
            x_valid_new = np.concatenate((x_valid_blur, pred_valid_blur), axis=-1)
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
        if not os.path.exists(os.path.join(input_dir, f'buffer_y_train_{dist}m.npy')): 
            buffers_y_train[dist] = buffer_patches(y_train, dist_cells=dist)
            np.save(input_dir + f'buffer_y_train_{dist}m.npy', buffers_y_train[dist])
            
        if not os.path.exists(os.path.join(input_dir, f'buffer_y_valid_{dist}m.npy')): 
            buffers_y_valid[dist] = buffer_patches(y_valid, dist_cells=dist)
            np.save(input_dir + f'buffer_y_valid_{dist}m.npy', buffers_y_valid[dist])
        
        # Buffers para Sensibilidade Relaxada
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_train_{dist}m.npy')):
            buffers_pred_train[dist] = buffer_patches(pred_train, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_train_{dist}m.npy', buffers_pred_train[dist])
            
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_valid_{dist}m.npy')): 
            buffers_pred_valid[dist] = buffer_patches(pred_valid, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_valid_{dist}m.npy', buffers_pred_valid[dist])
    
    
    # Lê buffers de arrays de predição do Treinamento e Validação
    for dist in dist_buffers:
        buffers_y_train[dist] = np.load(input_dir + f'buffer_y_train_{dist}m.npy')
        buffers_y_valid[dist] = np.load(input_dir + f'buffer_y_valid_{dist}m.npy')
        buffers_pred_train[dist] = np.load(output_dir + f'buffer_pred_train_{dist}m.npy')
        buffers_pred_valid[dist] = np.load(output_dir + f'buffer_pred_valid_{dist}m.npy')
        
        
    # Relaxed Metrics for training
    relaxed_precision_train, relaxed_recall_train, relaxed_f1score_train = {}, {}, {}
    relaxed_precision_valid, relaxed_recall_valid, relaxed_f1score_valid = {}, {}, {}
    
    
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}m.txt', 'a') as f:
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
    if not os.path.exists(os.path.join(output_dir, 'pred_test.npy')) or \
       not os.path.exists(os.path.join(output_dir, 'prob_test.npy')):
        pred_test, prob_test = Test_Step(model, x_test, 2)
        
        prob_test = prob_test[..., 1:2] # Essa é a probabilidade de prever estrada (valor de pixel 1)
    
        # Converte para tipos que ocupam menos espaço
        pred_test = pred_test.astype(np.uint8)
        prob_test = prob_test.astype(np.float16)
        
        # Salva arrays de predição do Teste. Arquivos da Predição (pred) são salvos na pasta de arquivos de saída (resultados_dir)
        salva_arrays(output_dir, pred_test=pred_test, prob_test=prob_test)
        
        
        
    # Lê arrays de predição do Teste
    pred_test = np.load(output_dir + 'pred_test.npy')
    prob_test = np.load(output_dir + 'prob_test.npy')
    
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
            pred_test_blur, _ = Test_Step(model, x_test_blur, 2)
            x_test_new = np.concatenate((x_test_blur, pred_test_blur), axis=-1)
            salva_arrays(output_dir, x_test=x_test_new)
        
    # Faz os Buffers necessários, para teste, nas imagens
    
    buffers_y_test = {}
    buffers_pred_test = {}
    
    for dist in dist_buffers:
        # Buffer para Precisão Relaxada
        if not os.path.exists(os.path.join(input_dir, f'buffer_y_test_{dist}m.npy')):
            buffers_y_test[dist] = buffer_patches(y_test, dist_cells=dist)
            np.save(input_dir + f'buffer_y_test_{dist}m.npy', buffers_y_test[dist])
            
        # Buffer para Sensibilidade Relaxada
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_test_{dist}m.npy')):
            buffers_pred_test[dist] = buffer_patches(pred_test, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_test_{dist}m.npy', buffers_pred_test[dist])
            
            
    # Lê buffers de arrays de predição do Teste
    for dist in dist_buffers:
        buffers_y_test[dist] = np.load(input_dir + f'buffer_y_test_{dist}m.npy')
        buffers_pred_test[dist] = np.load(output_dir + f'buffer_pred_test_{dist}m.npy')
        
        
    # Relaxed Metrics for test
    relaxed_precision_test, relaxed_recall_test, relaxed_f1score_test = {}, {}, {}
    
    
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}m.txt', 'a') as f:
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
        if not os.path.exists(os.path.join(input_dir, f'buffers_y_mosaics_{dist}m.npy')):
            buffers_y_mosaics[dist] = buffer_patches(y_mosaics, dist_cells=dist)
            np.save(input_dir + f'buffer_y_mosaics_{dist}m.npy', buffers_y_mosaics[dist])
            
        # Buffer dos Mosaicos de Predição   
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_mosaics_{dist}m.npy')):
            buffers_pred_mosaics[dist] = buffer_patches(pred_mosaics, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_mosaics_{dist}m.npy', buffers_pred_mosaics[dist])
    
    
    # Lê buffers de Mosaicos
    for dist in dist_buffers:
        buffers_y_mosaics[dist] = np.load(input_dir + f'buffer_y_mosaics_{dist}m.npy')
        buffers_pred_mosaics[dist] = np.load(output_dir + f'buffer_pred_mosaics_{dist}m.npy')  
        
        
    # Avaliação da Qualidade para os Mosaicos de Teste
    # Relaxed Metrics for mosaics test
    relaxed_precision_mosaics, relaxed_recall_mosaics, relaxed_f1score_mosaics = {}, {}, {}
      
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}m.txt', 'a') as f:
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









# Avalia um ensemble de modelos segundo conjuntos de treino, validação, teste e mosaicos de teste
# Retorna as métricas de precisão, recall e f1-score relaxados
# Além disso constroi os mosaicos de resultado 
# Etapa 1 se refere ao processamento no qual a entrada do pós-processamento é a predição do modelo
# Etapa 2 ao processamento no qual a entrada do pós-processamento é a imagem original mais a predição do modelo
# Etapa 3 ao processamento no qual a entrada do pós-processamento é a imagem original com blur gaussiano mais a predição desses dados ruidosos com o modelo
# Etapa 4 se refere ao pós-processamento   
def avalia_modelo_ensemble(input_dir: str, output_dir: str, metric_name = 'F1-Score', 
                           etapa=3, dist_buffers = [3], std_blur = 0.4):
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
    n_members = 5
    
    # Show and save history
    for i in range(n_members):
        with open(output_dir + 'history_pickle_' + best_model_filename + '_' + str(i+1) + '.pickle', "rb") as fp:   
            history = pickle.load(fp)
            
        # Show and save history
        show_graph_loss_accuracy(np.asarray(history), 1, metric_name = metric_name, save=True, save_path=output_dir,
                                 save_name='plotagem' + '_' + str(i) + '.png')
    
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
        pred_train = calcula_pred_from_prob_ensemble(prob_train_ensemble)
        pred_valid = calcula_pred_from_prob_ensemble(prob_valid_ensemble)
        
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
            pred_train_blur = calcula_pred_from_prob_ensemble(prob_train_ensemble_blur)
            pred_valid_blur = calcula_pred_from_prob_ensemble(prob_valid_ensemble_blur)
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
        if not os.path.exists(os.path.join(input_dir, f'buffer_y_train_{dist}m.npy')): 
            buffers_y_train[dist] = buffer_patches(y_train, dist_cells=dist)
            np.save(input_dir + f'buffer_y_train_{dist}m.npy', buffers_y_train[dist])
            
        if not os.path.exists(os.path.join(input_dir, f'buffer_y_valid_{dist}m.npy')): 
            buffers_y_valid[dist] = buffer_patches(y_valid, dist_cells=dist)
            np.save(input_dir + f'buffer_y_valid_{dist}m.npy', buffers_y_valid[dist])
        
        # Buffers para Sensibilidade Relaxada
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_train_{dist}m.npy')):
            buffers_pred_train[dist] = buffer_patches(pred_train, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_train_{dist}m.npy', buffers_pred_train[dist])
            
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_valid_{dist}m.npy')): 
            buffers_pred_valid[dist] = buffer_patches(pred_valid, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_valid_{dist}m.npy', buffers_pred_valid[dist])
    
    
    # Lê buffers de arrays de predição do Treinamento e Validação
    for dist in dist_buffers:
        buffers_y_train[dist] = np.load(input_dir + f'buffer_y_train_{dist}m.npy')
        buffers_y_valid[dist] = np.load(input_dir + f'buffer_y_valid_{dist}m.npy')
        buffers_pred_train[dist] = np.load(output_dir + f'buffer_pred_train_{dist}m.npy')
        buffers_pred_valid[dist] = np.load(output_dir + f'buffer_pred_valid_{dist}m.npy')
    
    
    # Relaxed Metrics for training and validation
    relaxed_precision_train, relaxed_recall_train, relaxed_f1score_train = {}, {}, {}
    relaxed_precision_valid, relaxed_recall_valid, relaxed_f1score_valid = {}, {}, {}
    
    
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}m.txt', 'a') as f:
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
        pred_test = calcula_pred_from_prob_ensemble(prob_test_ensemble)
        
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
            pred_test_blur = calcula_pred_from_prob_ensemble(prob_test_ensemble_blur)
            x_test_new = np.concatenate((x_test_blur, pred_test_blur), axis=-1).astype(np.float16)
            salva_arrays(output_dir, x_test=x_test_new)
    
    
    # Faz os Buffers necessários, para teste, nas imagens
    buffers_y_test = {}
    buffers_pred_test = {}
    
    for dist in dist_buffers:
        # Buffer para Precisão Relaxada
        if not os.path.exists(os.path.join(input_dir, f'buffer_y_test_{dist}m.npy')):
            buffers_y_test[dist] = buffer_patches(y_test, dist_cells=dist)
            np.save(input_dir + f'buffer_y_test_{dist}m.npy', buffers_y_test[dist])
            
        # Buffer para Sensibilidade Relaxada
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_test_{dist}m.npy')):
            buffers_pred_test[dist] = buffer_patches(pred_test, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_test_{dist}m.npy', buffers_pred_test[dist])
            
            
    # Lê buffers de arrays de predição do Teste
    for dist in dist_buffers:
        buffers_y_test[dist] = np.load(input_dir + f'buffer_y_test_{dist}m.npy')
        buffers_pred_test[dist] = np.load(output_dir + f'buffer_pred_test_{dist}m.npy')
    
    # Relaxed Metrics for test
    relaxed_precision_test, relaxed_recall_test, relaxed_f1score_test = {}, {}, {}
    
    
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}m.txt', 'a') as f:
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
        if not os.path.exists(os.path.join(input_dir, f'buffers_y_mosaics_{dist}m.npy')):
            buffers_y_mosaics[dist] = buffer_patches(y_mosaics, dist_cells=dist)
            np.save(input_dir + f'buffer_y_mosaics_{dist}m.npy', buffers_y_mosaics[dist])
            
        # Buffer dos Mosaicos de Predição   
        if not os.path.exists(os.path.join(output_dir, f'buffer_pred_mosaics_{dist}m.npy')):
            buffers_pred_mosaics[dist] = buffer_patches(pred_mosaics, dist_cells=dist)
            np.save(output_dir + f'buffer_pred_mosaics_{dist}m.npy', buffers_pred_mosaics[dist])
    
    
    # Lê buffers de Mosaicos
    for dist in dist_buffers:
        buffers_y_mosaics[dist] = np.load(input_dir + f'buffer_y_mosaics_{dist}m.npy')
        buffers_pred_mosaics[dist] = np.load(output_dir + f'buffer_pred_mosaics_{dist}m.npy')  
    
    # Avaliação da Qualidade para os Mosaicos de Teste
    # Relaxed Metrics for mosaics test
    relaxed_precision_mosaics, relaxed_recall_mosaics, relaxed_f1score_mosaics = {}, {}, {}
      
    for dist in dist_buffers:
        with open(output_dir + f'relaxed_metrics_{dist}m.txt', 'a') as f:
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
def calcula_pred_from_prob_ensemble(prob_array):
    # Calcula probabilidade média    
    prob_mean = np.mean(prob_array, axis=0)

    # Calcula predição (é a classe que, entre as 2 probabilidades, tem probabilidade maior)
    predicted_classes = np.argmax(prob_mean, axis=-1)[..., np.newaxis] 
    
    return predicted_classes



# pred_array é o array de predição e labels_paths é uma lista com o nome dos caminhos dos tiles
# que deram origem aos patches e que serão usados para informações de georreferência
# patch_test_stride é o stride com o qual foram extraídos os patches, necessário para a reconstrução dos tiles
# labels_test_shape é a dimensão de cada tile, também necessário para suas reconstruções
# n_test_tiles é a quantidade de tiles a serem reconstruídos
def gera_mosaicos(output_dir, pred_array, labels_paths, prefix='outmosaic', patch_test_stride=96, labels_test_shape=(1408, 1280), n_test_tiles=10, is_float=False):
    # Stride e Dimensões do Tile
    patch_test_stride = patch_test_stride # tem que estabelecer de forma manual de acordo com o stride usado para extrair os patches
    labels_test_shape = labels_test_shape # também tem que estabelecer de forma manual de acordo com as dimensões da imagem de referência
    
    # Lista com os mosaicos previstos
    pred_test_mosaic_list = []
    
    n_test_tiles = n_test_tiles
    n_patches_tile = int(pred_array.shape[0]/n_test_tiles) # Quantidade de patches por tile. Supõe tiles com igual número de patches
    n = len(pred_array) # Quantidade de patches totais, contando todos os tiles, é igual ao comprimento 
    
    i = 0 # Índice onde o tile começa
    i_mosaic = 0 # Índice que se refere ao número do mosaico (primeiro, segundo, terceiro, ...)
    
    while i < n:
        print('Making Mosaic {}/{}'.format(i_mosaic+1, n_test_tiles ))
        
        pred_test_mosaic = unpatch_reference(pred_array[i:i+n_patches_tile, ..., 0], patch_test_stride, labels_test_shape, border_patches=True)
        
        pred_test_mosaic_list.append(pred_test_mosaic) 
        
        # Incremento
        i = i+n_patches_tile
        i_mosaic += 1  
        
        
    # Salva mosaicos
    labels_paths = labels_paths
    output_dir = output_dir
    
    for i in range(len(pred_test_mosaic_list)):
        pred_test_mosaic = pred_test_mosaic_list[i]
        labels_path = labels_paths[i]
        
        filename_wo_ext = labels_path.split(os.path.sep)[-1].split('.tif')[0]
        tile_line = int(filename_wo_ext.split('_')[1])
        tile_col = int(filename_wo_ext.split('_')[2])
        
        out_mosaic_name = prefix + '_' + str(tile_line) + '_' + str(tile_col) + r'.tif'
        out_mosaic_path = os.path.join(output_dir, out_mosaic_name)
        
        if is_float:
            save_raster_reference(labels_path, out_mosaic_path, pred_test_mosaic, is_float=True)
        else:
            save_raster_reference(labels_path, out_mosaic_path, pred_test_mosaic, is_float=False)
            
            
    # Retorna lista de mosaicos
    return pred_test_mosaic_list
            