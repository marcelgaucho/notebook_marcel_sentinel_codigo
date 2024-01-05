# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:07:15 2023

@author: marcel.rotunno
"""

from osgeo import gdal

import tensorflow as tf 
import os
import pdb

# Desabilita GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#tf.config.threading.set_inter_op_parallelism_threads(2)
#tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# %%

from functions_bib import avalia_modelo

# %% Teste da função de avaliação

input_dir = r'entrada/'
y_dir = r'y_directory/'
output_dir = r'saida_mnih_dice/'
print('Input Dir: ', input_dir)
print('Y Dir: ', y_dir)
print('Output Dir: ', output_dir)

#pdb.set_trace()
dicio_resultados = avalia_modelo(input_dir, y_dir, output_dir, metric_name = 'F1-Score',
                                 etapa=5, dist_buffers = [0, 3], avalia_train=True)