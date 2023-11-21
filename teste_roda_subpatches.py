# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:46:28 2023

@author: marcel.rotunno
"""


import os, shutil
import tensorflow as tf
import pickle

'''
# Limita memória da GPU 

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')

# Limita Memória usada pela GPU 
tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=12288)])
'''

from functions_bib import treina_modelo

# %%

model_dir = 'saida_mnih/'

 
# %% Função para fazer tudo isso

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

        

# %% Carrega dados da Preparação

with open(model_dir + 'filenames_train_list.pickle', "rb") as fp:   
    filenames_train_list = pickle.load(fp)
    
with open(model_dir + 'filenames_valid_list.pickle', "rb") as fp:   
    filenames_valid_list = pickle.load(fp)
    
with open(model_dir + 'filenames_test_list.pickle', "rb") as fp:   
    filenames_test_list = pickle.load(fp)

# %% Testa função de Rodar Treinamento

treina_com_subpatches(filenames_train_list, filenames_valid_list, filenames_test_list, model_dir)    

