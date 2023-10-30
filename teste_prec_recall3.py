# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:13:11 2023

@author: marcel.rotunno
"""

from osgeo import gdal
import numpy as np

import tensorflow as tf 
import os

# Desabilita GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#tf.config.threading.set_inter_op_parallelism_threads(2)
#tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

from functions_bib import plota_curva_precision_recall_relaxada

# %% Carrega dados

input_dir = 'entrada/'
output_dir = 'saida_mnih/'

y_test = np.load(input_dir + 'y_test.npy')
prob_test = np.load(output_dir + 'prob_test.npy')
buffer_y_test_3px = np.load(input_dir + 'buffer_y_test_3px.npy')

# %% Executa função


precision_scores, recall_scores, intersection = plota_curva_precision_recall_relaxada(y_test, prob_test,
                                                                                      buffer_y_test_3px, buffer_px=3, num_pontos=10,
                                                                                      output_dir=output_dir, 
                                                                                      save_figure=True)


