# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:07:15 2023

@author: marcel.rotunno
"""

from osgeo import gdal

import tensorflow as tf 
import os

# Desabilita GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#tf.config.threading.set_inter_op_parallelism_threads(2)
#tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(3)
tf.config.threading.set_intra_op_parallelism_threads(3)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# %%

from functions_bib import avalia_modelo

# %% Teste da função de avaliação

input_dir = 'early_cross_f1_chamorro/'
output_dir = 'early_100_cross_f1_chamorro_pos/'

dicio_resultados = avalia_modelo(input_dir, output_dir, metric_name = 'F1-Score',
                                 etapa=5, dist_buffers = [0, 1], std_blur = 1, subpatch_size=8, k=20)