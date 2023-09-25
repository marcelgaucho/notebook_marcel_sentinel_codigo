# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:05:37 2023

@author: marcel.rotunno
"""

import numpy as np
import tensorflow as tf


# Limita memória da GPU 

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')

# Limita Memória usada pela GPU 
tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=12288)])



# %% Diretórios de entrada e saída

input_dir = 'early_cross_f1_chamorro/'
output_dir = 'early_100_cross_f1_chamorro_pos_curta/'


# %%  Importa função
from functions_bib import treina_modelo

 
# %% Teste da Função

#tf.config.run_functions_eagerly(False)
treina_modelo(input_dir, output_dir, epochs=2000, early_loss=False, 
              model_type='resunet chamorro pos curta', loss='cross')
best_model_filename = 'best_model'
