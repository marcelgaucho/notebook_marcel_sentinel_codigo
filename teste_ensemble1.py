# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:06:46 2023

@author: marcel.rotunno
"""

import numpy as np
import tensorflow as tf


# %% Diretórios de entrada e saída

input_dir = 'entrada/'
output_dir = 'focal_early_f1/'


# %%  Importa função

from functions_bib import treina_modelo_ensemble

# %% Teste da Função

treina_modelo_ensemble(input_dir, output_dir, n_members=5, epochs=150, early_loss=True, metric='accuracy')