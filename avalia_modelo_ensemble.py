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
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# %%

from functions_bib import avalia_modelo_ensemble

# %% Teste da função de avaliação

input_dir = r'early_f1_focal_1000_chamorro/'
output_dir = r'early_f1_focal_1000_chamorro_correction_with_image_1000_chamorro/'

dicio_resultados = avalia_modelo_ensemble(input_dir, output_dir, metric_name = 'F1-Score',
                                 etapa=4, dist_buffers = [1,2,3], std_blur = 0.4, n_members=5)