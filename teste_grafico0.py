# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:33:38 2023

@author: marcel.rotunno
"""
# Importa função
from functions_bib import gera_graficos

# Diretórios e nomes dos experimentos
apresentacao_dir = 'graficos_apresentacao/'
metrics_dirs_list = ['early_f1_cross_500/', 'early_f1_focal_500/', 'early_f1_cross_500_chamorro/', 
                     'early_f1_focal_500_chamorro/']
lista_nomes_exp = ['Early F1 Cross 500', 'Early F1 Focal 500', 'Early F1 Cross 500 Chamorro', 
                   'Early F1 Focal 500 Chamorro']

# %% Gera os gráficos

# Gera gráficos
gera_graficos(metrics_dirs_list, lista_nomes_exp, save_path=apresentacao_dir)

