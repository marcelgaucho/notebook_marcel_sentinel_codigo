# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:46:28 2023

@author: marcel.rotunno
"""

from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import gc
import os
import tensorflow as tf
import pickle

# Desabilita GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#tf.config.threading.set_inter_op_parallelism_threads(2)
#tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



'''
# Limita memória da GPU 

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')

# Limita Memória usada pela GPU 
tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=12288)])
'''

# %%

from functions_bib import blur_x_patches, salva_arrays, treina_modelo, Test_Step
from tensorflow.keras.models import load_model 
import shutil

# %%

input_dir = 'entrada/'
model_dir = 'saida_mnih/'





 
# %% Função para fazer tudo isso

def prepara_subpatches(input_dir, model_dir):
    # Carrega Xs e Ys
    x_train = np.load(input_dir + 'x_train.npy')
    x_valid = np.load(input_dir + 'x_valid.npy')
    x_test = np.load(input_dir + 'x_test.npy')
    y_train = np.load(input_dir + 'y_train.npy')
    y_valid = np.load(input_dir + 'y_valid.npy')
    y_test = np.load(input_dir + 'y_test.npy')
    # Carrega modelo
    best_model_filename = 'best_model'
    model_path = model_dir + best_model_filename + '.h5'
    model = load_model(model_path, compile=False)
    # Cria diretório para guardar os Xs modificados
    subpatches_dir = os.path.join(model_dir, 'subpatches1')
    os.makedirs(subpatches_dir, exist_ok=True)
    # Forma dados com subpatches
    patch_size = 224
    #subpatch_sizes = [32, 16, 8, 4]
    #std_blur_dict = {'fraco':0.4, 'médio':0.6, 'forte': 0.8}
    subpatch_sizes = [28]
    #std_blur_dict = {'forte':2, 'muito forte': 3}
    std_blur_dict = {'muito forte': 3}
    ks = [5, 10, 15, 20]    
    filenames_train_list = []
    filenames_valid_list = []
    filenames_test_list = []
    
    # Previsão do Teste é Como a Pós Normal 
    # Está Fora do loop, pois para todas as variações de subpatches são iguais,
    # já que somente envolve x_test e pred_test, não envolve blur
    pred_test, _ = Test_Step(model, x_test, 2)
    
    # Concatenate with original to form new x_test
    x_test_new = np.concatenate((x_test, pred_test), axis=-1).astype(np.float16)
    
    del x_test, pred_test
    gc.collect()     
    
    # Loop para geração dos subpatches
    for subpatch_size in subpatch_sizes:
        for std_blur in std_blur_dict.values():
            for k in ks:
                # Generate Dirs
                dirname = f'dim{subpatch_size}_stdblur{std_blur}_k{k}'
                dirname_extended = os.path.join(subpatches_dir, dirname)
                os.makedirs(dirname_extended, exist_ok=True)
                
                # Make X for X_Train
                x_train_blur, pred_train_blur = blur_x_patches(x_train, y_train, dim=subpatch_size, k=k, blur=std_blur, model=model)
                x_train_new = np.concatenate((x_train_blur, pred_train_blur), axis=-1).astype(np.float16)
                salva_arrays(dirname_extended, x_train=x_train_new)
                
                del x_train_blur, pred_train_blur, x_train_new
                gc.collect()

                # Make X for X_Valid
                x_valid_blur, pred_valid_blur = blur_x_patches(x_valid, y_valid, dim=subpatch_size, k=k, blur=std_blur, model=model)
                x_valid_new = np.concatenate((x_valid_blur, pred_valid_blur), axis=-1).astype(np.float16)
                salva_arrays(dirname_extended, x_valid=x_valid_new)
                
                del x_valid_blur, pred_valid_blur, x_valid_new
                gc.collect() 
                
                # Save New Xs for Test
                salva_arrays(dirname_extended, x_test=x_test_new)
                
                # Append to file names list
                filenames_train_list.append(os.path.join(dirname_extended, 'x_train.npy'))
                filenames_valid_list.append(os.path.join(dirname_extended, 'x_valid.npy'))
                filenames_test_list.append(os.path.join(dirname_extended, 'x_test.npy'))
                
        
    return filenames_train_list, filenames_valid_list, filenames_test_list


    

# %% Testa função de Preparação dos Subpatches

filenames_train_list, filenames_valid_list, filenames_test_list = prepara_subpatches(input_dir, model_dir)

with open(model_dir + 'filenames_train_list' +  '.pickle', "wb") as fp: 
    pickle.dump(filenames_train_list, fp)
    
with open(model_dir + 'filenames_valid_list' +  '.pickle', "wb") as fp: 
    pickle.dump(filenames_valid_list, fp)
    
with open(model_dir + 'filenames_test_list' +  '.pickle', "wb") as fp: 
    pickle.dump(filenames_test_list, fp)

  

