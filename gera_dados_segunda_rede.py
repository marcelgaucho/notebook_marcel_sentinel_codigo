# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:07:15 2023

@author: marcel.rotunno
"""

import h5py
from osgeo import gdal

import tensorflow as tf 
import os

# Desabilita GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#tf.config.threading.set_inter_op_parallelism_threads(2)
#tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# %%

from functions_bib import gera_dados_segunda_rede

# %% Teste da função de avaliação

input_dir = r'entrada/'
y_dir = r'y_directory/'
output_dir = r'saida_mnih/'

gera_dados_segunda_rede(input_dir, y_dir, output_dir, etapa=3)