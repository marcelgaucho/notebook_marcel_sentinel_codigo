# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:47:11 2023

@author: marcel.rotunno
"""

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

from functions_bib import avalia_modelo

# %% Diretórios

subpatches_dir = r'saida_mnih/subpatches/'

# %%

dirs_subpatches = [os.path.join(subpatches_dir, name) + r'/' for name in os.listdir(subpatches_dir) if os.path.isdir(os.path.join(subpatches_dir, name))]

# %% Roda avaliação de modelo para cada subdiretório dentro de subpatches

for dir_sub in dirs_subpatches:
    avalia_modelo(dir_sub, dir_sub, metric_name = 'F1-Score',
                  etapa=5, dist_buffers = [0, 3])