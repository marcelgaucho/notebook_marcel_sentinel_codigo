# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 19:01:23 2023

@author: marce
"""

# Importação das bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import os
import math
from functions_bib import load_tiff_image, load_tiff_image_reference, normalization
from functions_bib import extract_patches, salva_arrays, train_unet, build_model
from functions_bib import show_graph_loss_accuracy, compute_relaxed_metrics, unpatch_reference
from functions_bib import buffer_patches, save_raster_reference, Test_Step
from functions_bib import extract_tiles, filtra_tiles_estradas, copia_tiles_filtrados, extract_patches_from_tiles




# Pasta onde os modelos vão ser lidos/escritos
if not os.path.exists('modelos'):
    os.mkdir('modelos')

root_path = 'modelos/'

# Images Directories
train_test_dir = 'entrada1/'
resultados_dir = 'resultados1/'


# %% Extrai patches (Info)

# Tamanho do patch. Patch é quadrado (altura=largura)
patch_size = 128

# Stride do Patch. Com um stride menor que a largura do patch há sobreposição entre os patches
# 25% de sobreposição entre patches
patch_stride = patch_size - (patch_size // 4)

# Número de canais da imagem/patch
image_channels = 4

# Dimensões do patch
input_shape = (patch_size, patch_size, image_channels)




# %% Varre diretório de treino e de validação para abrir os tiles dos quais os patches serão extraídos

train_imgs_tiles_dir = r'new_teste_tiles\imgs\train'
valid_imgs_tiles_dir = r'new_teste_tiles\imgs\valid'
train_labels_tiles_dir = r'new_teste_tiles\masks\train'
valid_labels_tiles_dir = r'new_teste_tiles\masks\valid'
test_imgs_tiles_dir = r'new_teste_tiles\imgs\test'
test_labels_tiles_dir = r'new_teste_tiles\masks\test'
#newroads_valid_labels_tiles_dir = r'new_teste_tiles\masks\valid_new_roads'
newroads_test_labels_tiles_dir = r'new_teste_tiles\masks\test_new_roads'

train_imgs_tiles = [os.path.join(train_imgs_tiles_dir, arq) for arq in os.listdir(train_imgs_tiles_dir)]
valid_imgs_tiles = [os.path.join(valid_imgs_tiles_dir, arq) for arq in os.listdir(valid_imgs_tiles_dir)]
test_imgs_tiles = [os.path.join(test_imgs_tiles_dir, arq) for arq in os.listdir(test_imgs_tiles_dir)]

train_labels_tiles = [os.path.join(train_labels_tiles_dir, arq) for arq in os.listdir(train_labels_tiles_dir)]
valid_labels_tiles = [os.path.join(valid_labels_tiles_dir, arq) for arq in os.listdir(valid_labels_tiles_dir)]
test_labels_tiles = [os.path.join(test_labels_tiles_dir, arq) for arq in os.listdir(test_labels_tiles_dir)]
#newroads_valid_labels_tiles = [os.path.join(newroads_valid_labels_tiles_dir, arq) for arq in os.listdir(newroads_valid_labels_tiles_dir)]
newroads_test_labels_tiles = [os.path.join(newroads_test_labels_tiles_dir, arq) for arq in os.listdir(newroads_test_labels_tiles_dir)]


train_imgs_labels_dict = dict(zip(train_imgs_tiles, train_labels_tiles))
valid_imgs_labels_dict = dict(zip(valid_imgs_tiles, valid_labels_tiles))
test_imgs_labels_dict = dict(zip(test_imgs_tiles, test_labels_tiles))
#newroads_valid_labels_dict = dict(zip(valid_imgs_tiles, newroads_valid_labels_tiles)) # Tanto faz colocar valid_imgs_tiles ou test_imgs_tiles,
                                                                                      # porque os tiles das imagens serão descartados, já que eles terão
                                                                                      # sido criados de qualquer maneira
newroads_test_labels_dict = dict(zip(test_imgs_tiles, newroads_test_labels_tiles))





# %% Extrai patches de treino


x_train, y_train = extract_patches_from_tiles(train_imgs_labels_dict, patch_size, patch_stride)

x_valid, y_valid = extract_patches_from_tiles(valid_imgs_labels_dict, patch_size, patch_stride)

x_test, y_test = extract_patches_from_tiles(test_imgs_labels_dict, patch_size, patch_stride, border_patches=True)

#_, newroads_y_valid = extract_patches_from_tiles(newroads_valid_labels_dict, patch_size, patch_stride, border_patches=True)

_, newroads_y_test = extract_patches_from_tiles(newroads_test_labels_dict, patch_size, patch_stride, border_patches=True)


# %% Teste com 1 tile apenas

test_imgs_labels_dict0 = {r'new_teste_tiles\\imgs\\test\\tile_0_2.tif': r'new_teste_tiles\\masks\\test\\reftile_0_2.tif'}

x_test0, y_test0 = extract_patches_from_tiles(test_imgs_labels_dict0, patch_size, patch_stride, border_patches=True)

# %% Filtra Patches que contenham mais que 1% de pixels de estrada

x_train, y_train, indices_maior_train = filtra_tiles_estradas(x_train, y_train, 1)

x_valid, y_valid, indices_maior_valid = filtra_tiles_estradas(x_valid, y_valid, 1)

#x_test, y_test, indices_maior_test = filtra_tiles_estradas(x_valid, y_valid, 0)


# %% Faz aumento de dados (Função)

def augment_data(images: np.ndarray) -> np.ndarray:
    augmented_images = []
    for image in images:
        # Rotação 90 graus
        augmented_images.append(np.rot90(image, 1))
        # Rotação 180 graus
        augmented_images.append(np.rot90(image, 2))
        # Rotação 270 graus
        augmented_images.append(np.rot90(image, 3))
        # Flip horizontal
        augmented_images.append(np.fliplr(image))
        # Flip vertical
        augmented_images.append(np.flipud(image))
    
    return np.stack(augmented_images)


# Faz aumento de dados para arrays x e y na forma
# (batches, heigth, width, channels)
# Rotações sentido anti-horário 90, 180, 270
# Espelhamento Vertical e Horizontal
def aumento_dados(x, y):
    # Rotações Sentido Anti-Horário
    # Rotação 90 graus
    x_rot90 = np.rot90(x, k=1, axes=(1,2))
    y_rot90 = np.rot90(y, k=1, axes=(1,2))
    # Rotação 180 graus 
    x_rot180 = np.rot90(x, k=2, axes=(1,2))
    y_rot180 = np.rot90(y, k=2, axes=(1,2))
    # Rotação 270 graus 
    x_rot270 = np.rot90(x, k=3, axes=(1,2))
    y_rot270 = np.rot90(y, k=3, axes=(1,2))
    
    # Espelhamento Vertical (Mirror)
    x_mirror = np.flip(x, axis=2)
    y_mirror = np.flip(y, axis=2)
    
    # Espelhamento Horizontal (Flip)
    x_flip = np.flip(x, axis=1)
    y_flip = np.flip(y, axis=1)

    x_aum = np.concatenate((x, x_rot90, x_rot180, x_rot270, x_mirror, x_flip))
    y_aum = np.concatenate((y, y_rot90, y_rot180, y_rot270, y_mirror, y_flip))

    return x_aum, y_aum                
    
    
    

# %% Faz aumento de dados

x_train, y_train = aumento_dados(x_train, y_train)

#x_valid, y_valid = aumento_dados(x_valid, y_valid)


# %% Seleciona Patches que contenham estrada

# Exemplo de um patch de treinamento que contenha estrada
image_index = 0

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_train[image_index, :, :, 0:3]))
ax1.set_title('Train Patch', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_train[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Train Labeled reference', fontsize=20)
ax2.axis('off')


# Exemplo de um patch de treinamento que contenha estrada (Rotacionado 90 graus)
image_index = 1430

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_train[image_index, :, :, 0:3]))
ax1.set_title('Train Patch (Rotated 90 degrees)', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_train[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Train Labeled reference (Rotated 90 degrees)', fontsize=20)
ax2.axis('off')


# Exemplo de um patch de treinamento que contenha estrada (Rotacionado 180 graus)
image_index = 2860

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_train[image_index, :, :, 0:3]))
ax1.set_title('Train Patch (Rotated 180 degrees)', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_train[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Train Labeled reference (Rotated 180 degrees)', fontsize=20)
ax2.axis('off')


# Exemplo de um patch de treinamento que contenha estrada (Rotacionado 270 graus)
image_index = 4290

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_train[image_index, :, :, 0:3]))
ax1.set_title('Train Patch (Rotated 270 degrees)', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_train[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Train Labeled reference (Rotated 270 degrees)', fontsize=20)
ax2.axis('off')


# Exemplo de um patch de treinamento que contenha estrada (Espelhamento Vertical)
image_index = 5720

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_train[image_index, :, :, 0:3]))
ax1.set_title('Train Patch (Espelhamento Vertical)', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_train[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Train Labeled reference (Espelhamento Vertical)', fontsize=20)
ax2.axis('off')

# Exemplo de um patch de treinamento que contenha estrada (Espelhamento Horizontal)
image_index = 7150

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_train[image_index, :, :, 0:3]))
ax1.set_title('Train Patch (Espelhamento Horizontal)', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_train[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Train Labeled reference (Espelhamento Horizontal)', fontsize=20)
ax2.axis('off')        


# Exemplo de um patch de validação que contenha estrada
image_index = 0

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_valid[image_index, :, :, 0:3]))
ax1.set_title('Valid Patch', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_valid[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Valid Labeled reference', fontsize=20)
ax2.axis('off')


# Exemplo de um patch de validação que contenha estrada (Rotacionado 90 graus)
image_index = 358

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_valid[image_index, :, :, 0:3]))
ax1.set_title('Valid Patch (Rotated 90 degrees)', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_valid[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Valid Labeled reference (Rotated 90 degrees)', fontsize=20)
ax2.axis('off')


# Exemplo de um patch de validação que contenha estrada (Rotacionado 180 graus)
image_index = 716

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_valid[image_index, :, :, 0:3]))
ax1.set_title('Valid Patch (Rotated 180 degrees)', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_valid[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Valid Labeled reference (Rotated 180 degrees)', fontsize=20)
ax2.axis('off')


# Exemplo de um patch de validação que contenha estrada (Rotacionado 270 graus)
image_index = 1074

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_valid[image_index, :, :, 0:3]))
ax1.set_title('Valid Patch (Rotated 270 degrees)', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_valid[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Valid Labeled reference (Rotated 270 degrees)', fontsize=20)
ax2.axis('off')


# Exemplo de um patch de validação que contenha estrada (Espelhamento Vertical)
image_index = 1432

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_valid[image_index, :, :, 0:3]))
ax1.set_title('Valid Patch (Espelhamento Vertical)', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_valid[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Valid Labeled reference (Espelhamento Vertical)', fontsize=20)
ax2.axis('off')


# Exemplo de um patch de validação que contenha estrada (Espelhamento Vertical)
image_index = 1790

print('\nPlotting images...')
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 10))

ax1.imshow( normalization(x_valid[image_index, :, :, 0:3]))
ax1.set_title('Valid Patch (Espelhamento Horizontal)', fontsize=20)
ax1.axis('off')
 
#ax2.imshow( y_patches[image_index, :, :, 0], cmap='gray', vmin=0, vmax=num_classes )
ax2.imshow( y_valid[image_index, :, :, :], cmap='gray', interpolation='nearest')
ax2.set_title('Valid Labeled reference (Espelhamento Horizontal)', fontsize=20)
ax2.axis('off')       

# %% Salva dados de treino e validação para poderem ser usados por outro script

x_train = x_train.astype(np.float16)
x_valid = x_valid.astype(np.float16)

salva_arrays(train_test_dir, x_train=x_train, y_train=y_train, 
             x_valid=x_valid, y_valid=y_valid)

# %% Salva dados de teste para serem usados em outro script 

x_test = x_test.astype(np.float16)
y_test = y_test.astype(np.uint8)
newroads_y_test = newroads_y_test.astype(np.uint8)

salva_arrays(train_test_dir, x_test=x_test, y_test=y_test, newroads_y_test = newroads_y_test)


