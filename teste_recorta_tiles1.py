# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:29:15 2023

@author: marce
"""

# %% Preparação do Ambiente - 1

# Importação das bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array
import os, shutil
import math
from functions_bib import normalization, extract_tiles


# %% Abre e normaliza imagem para teste

# Abre como array numpy
imagem_treino = 'Image_2018.tif'
ds = gdal.Open(imagem_treino)

img_treino = ds.ReadAsArray()
img_treino = np.transpose(img_treino, (1,2,0))

# Referência de treinamento
referencia_treino = r'Train_RJ25_2016.tif'
ds = gdal.Open(referencia_treino)
ref_treino = ds.ReadAsArray()

# Referência de teste
referencia_teste = r'Test_RJ25_2018.tif'
ds = gdal.Open(referencia_teste)
ref_teste = ds.ReadAsArray()

# Referência de novas estradas de teste
referencia_teste = r'diff_estradas_2018_to_buffer_estradas_2016_10m_img_extent.tif'
ds = gdal.Open(referencia_teste)
ref_teste = ds.ReadAsArray()


# %% Processa imagem
gt = ds.GetGeoTransform()


# get coordinates of upper left corner
xmin = gt[0]
ymax = gt[3]
res = gt[1]

# determine total length of raster
xlen = res * ds.RasterXSize
ylen = res * ds.RasterYSize

# number of tiles in x and y direction
xdiv = 2
ydiv = 2

# size of a single tile
xsize = xlen/xdiv
ysize = ylen/ydiv

# create lists of x and y coordinates
xsteps = [xmin + xsize * i for i in range(xdiv+1)]
ysteps = [ymax - ysize * i for i in range(ydiv+1)]

# loop over min and max x and y coordinates
for i in range(xdiv):
    for j in range(ydiv):
        xmin = xsteps[i]
        xmax = xsteps[i+1]
        ymax = ysteps[j]
        ymin = ysteps[j+1]
        
        print("xmin: "+str(xmin))
        print("xmax: "+str(xmax))
        print("ymin: "+str(ymin))
        print("ymax: "+str(ymax))
        print("\n")
        
        # use gdal warp
        gdal.Warp("dem"+str(i)+str(j)+".tif", ds, 
                  outputBounds = (xmin, ymin, xmax, ymax), dstNodata = -9999)
        # or gdal translate to subset the input raster
        gdal.Translate("dem_translate"+str(i)+str(j)+".tif", ds, projWin = (xmin, ymax, xmax, ymin), xRes = res, yRes = -res)
 
# close the open dataset!!!
ds = None




# %% Função que extrai os tiles

def extract_tiles(input_img, input_ref, xoff, yoff, xsize, ysize, folder):
    # Open image 
    ds = gdal.Open(input_img)
    img_np = ds.ReadAsArray()
    img_np = np.transpose(img_np, (1,2,0))
    img_np = normalization(img_np)
    ds_ref = gdal.Open(input_ref)
    img_ref_np = ds_ref.ReadAsArray()
    
    # GeoTransform data
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = gdal_array.OpenArray(np.transpose(img_np, (2,0,1)).astype(np.float32)) # Update dataset to normalized image
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    
    print('proj img ', proj)
    print('Geotransform img ', gt)
    

    # Coordinates of upper left corner
    xmin = gt[0]
    ymax = gt[3]
    res = gt[1]
    
    xmin = xmin + res*xoff # Coordinates updated for offset
    ymax = ymax - res*yoff
    
    # Image from (xoff, yoff) to end of image (lower right)
    subimg_np = img_np[xoff:, yoff:, :]
    subimg_ref_np = img_ref_np[xoff:, yoff:]
    
    # Listas com os patches da imagem e da referência
    patch_img = []
    patch_ref = [] 
    
    # Altura e Largura percorrendo a imagem com o stride
    # No caso, onde não há sobreposição, o stride horizontal será igual 
    # a largura da imagem o stride vertical será igual a altura da imagem
    hstride = xsize
    vstride = ysize
    h = math.ceil(img_np.shape[0] / vstride)
    w = math.ceil(img_np.shape[1] / hstride)
    

    # Percorre dimensões obtidas com stride
    for m in range(0, h):
        for n in range(0, w):
            print('Processing tile ', m, n)
            # Índices relativos à altura e largura percorrendo com stride
            i_h = m*vstride
            i_w = n*hstride
            
            # Número de colunas e linhas do resultado (caso o patch não seja adicionado)
            if n == w-1: cols = w-1
            if m == h-1: lines = h-1
            
            # Adiciona Patch da Imagem e Referência caso ele esteja contido na imagem de entrada
            #print('M %d, N %d, Height start %d finish %d , Width start %d finish %d' % (m, n, i_h , i_h+patch_size, i_w, i_w+patch_size) )
            if ( (i_h + ysize <= img_np.shape[0]) and (i_w + xsize <= img_np.shape[1]) ):
                patch_img.append( subimg_np[i_h : i_h+ysize, i_w : i_w+xsize, :] )
                patch_ref.append( subimg_ref_np[i_h : i_h+ysize, i_w : i_w+xsize] )
                
                # Altera número de colunas e linhas do resultado (caso o patch seja adicionado)
                if n == w-1: cols = w
                if m == h-1: lines = h
                
                # Limites x e y do patch
                xmin_tile = xmin + i_w*res 
                xmax_tile = xmin_tile + hstride*res 
                ymax_tile = ymax - i_h*res
                ymin_tile = ymax_tile - vstride*res
                
                print(f'Limites do Patch xmin={xmin_tile}, xmax_tile={xmax_tile}, ymax_tile={ymax_tile}, ymin_tile={ymin_tile}')
                
                # Exporta tiles para pasta
                # Formato linha_coluna para tile (imagem) ou reftile (referência)
                
                tile_name = 'tile_' + str(m) + '_' + str(n) + '.tif'
                tile_ref_name = 'reftile_' + str(m) + '_' + str(n) + '.tif'
                
                gdal.Translate(os.path.join(folder, tile_name), ds, projWin = (xmin_tile, ymax_tile, xmax_tile, ymin_tile), 
                               xRes = res, yRes = -res)
                gdal.Translate(os.path.join(folder, tile_ref_name), ds_ref, projWin = (xmin_tile, ymax_tile, xmax_tile, ymin_tile), 
                               xRes = res, yRes = -res, noData=255)  
                
                
        
    # Retorna os arrays de patches da imagem e da referência        
    return np.array(patch_img), np.array(patch_ref), lines, cols


# %% Função que extrai os tiles somente da referência

def extract_tiles(input_img, xoff, yoff, xsize, ysize, folder):
    # Open image 
    ds = gdal.Open(input_img)
    img_np = ds.ReadAsArray()
    img_np = np.transpose(img_np, (1,2,0))
    img_np = normalization(img_np)
    ds_ref = gdal.Open(input_ref)
    img_ref_np = ds_ref.ReadAsArray()
    
    # GeoTransform data
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = gdal_array.OpenArray(np.transpose(img_np, (2,0,1)).astype(np.float32)) # Update dataset to normalized image
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    
    print('proj img ', proj)
    print('Geotransform img ', gt)
    

    # Coordinates of upper left corner
    xmin = gt[0]
    ymax = gt[3]
    res = gt[1]
    
    xmin = xmin + res*xoff # Coordinates updated for offset
    ymax = ymax - res*yoff
    
    # Image from (xoff, yoff) to end of image (lower right)
    subimg_np = img_np[xoff:, yoff:, :]
    subimg_ref_np = img_ref_np[xoff:, yoff:]
    
    # Listas com os patches da imagem e da referência
    patch_img = []
    patch_ref = [] 
    
    # Altura e Largura percorrendo a imagem com o stride
    # No caso, onde não há sobreposição, o stride horizontal será igual 
    # a largura da imagem o stride vertical será igual a altura da imagem
    hstride = xsize
    vstride = ysize
    h = math.ceil(img_np.shape[0] / vstride)
    w = math.ceil(img_np.shape[1] / hstride)
    

    # Percorre dimensões obtidas com stride
    for m in range(0, h):
        for n in range(0, w):
            print('Processing tile ', m, n)
            # Índices relativos à altura e largura percorrendo com stride
            i_h = m*vstride
            i_w = n*hstride
            
            # Número de colunas e linhas do resultado (caso o patch não seja adicionado)
            if n == w-1: cols = w-1
            if m == h-1: lines = h-1
            
            # Adiciona Patch da Imagem e Referência caso ele esteja contido na imagem de entrada
            #print('M %d, N %d, Height start %d finish %d , Width start %d finish %d' % (m, n, i_h , i_h+patch_size, i_w, i_w+patch_size) )
            if ( (i_h + ysize <= img_np.shape[0]) and (i_w + xsize <= img_np.shape[1]) ):
                patch_img.append( subimg_np[i_h : i_h+ysize, i_w : i_w+xsize, :] )
                patch_ref.append( subimg_ref_np[i_h : i_h+ysize, i_w : i_w+xsize] )
                
                # Altera número de colunas e linhas do resultado (caso o patch seja adicionado)
                if n == w-1: cols = w
                if m == h-1: lines = h
                
                # Limites x e y do patch
                xmin_tile = xmin + i_w*res 
                xmax_tile = xmin_tile + hstride*res 
                ymax_tile = ymax - i_h*res
                ymin_tile = ymax_tile - vstride*res
                
                print(f'Limites do Patch xmin={xmin_tile}, xmax_tile={xmax_tile}, ymax_tile={ymax_tile}, ymin_tile={ymin_tile}')
                
                # Exporta tiles para pasta
                # Formato linha_coluna para tile (imagem) ou reftile (referência)
                
                tile_name = 'tile_' + str(m) + '_' + str(n) + '.tif'
                tile_ref_name = 'reftile_' + str(m) + '_' + str(n) + '.tif'
                
                gdal.Translate(os.path.join(folder, tile_name), ds, projWin = (xmin_tile, ymax_tile, xmax_tile, ymin_tile), 
                               xRes = res, yRes = -res)
                gdal.Translate(os.path.join(folder, tile_ref_name), ds_ref, projWin = (xmin_tile, ymax_tile, xmax_tile, ymin_tile), 
                               xRes = res, yRes = -res, noData=255)  
                
                
        
    # Retorna os arrays de patches da imagem e da referência        
    return np.array(patch_img), np.array(patch_ref), lines, cols

# %% Extrai tiles

# Treinamento
img_tiles, ref_tiles, lines, cols = extract_tiles(imagem_treino, referencia_treino, 0, 0, 1280, 1408, 'teste2016_tiles')

# Teste
img_tiles, ref_tiles, lines, cols = extract_tiles(imagem_treino, referencia_teste, 0, 0, 1280, 1408, 'teste_tiles')

# Teste Diferença
img_tiles, ref_tiles, lines, cols = extract_tiles(imagem_treino, referencia_teste, 0, 0, 1280, 1408, 'testeDiff_tiles')


# %% Extrai tiles Debug

input_img = imagem_treino
input_ref = referencia_treino
xoff = 0
yoff = 0
xsize = 1280
ysize = 1408
folder = 'teste_tiles'


# Open image 
ds = gdal.Open(input_img)
img_np = ds.ReadAsArray()
img_np = np.transpose(img_np, (1,2,0))
img_np = normalization(img_np)
ds_ref = gdal.Open(input_ref)
img_ref_np = ds_ref.ReadAsArray()

# GeoTransform data
gt = ds.GetGeoTransform()
proj = ds.GetProjection()
ds = gdal_array.OpenArray(np.transpose(img_np, (2,0,1)).astype(np.float32)) # Update dataset to normalized image
ds.SetGeoTransform(gt)
ds.SetProjection(proj)

print('proj img ', proj)
print('Geotransform img ', gt)


# Coordinates of upper left corner
xmin = gt[0]
ymax = gt[3]
res = gt[1]

xmin = xmin + res*xoff # Coordinates updated for offset
ymax = ymax - res*yoff

# Image from (xoff, yoff) to end of image (lower right)
subimg_np = img_np[xoff:, yoff:, :]
subimg_ref_np = img_ref_np[xoff:, yoff:]

# Listas com os patches da imagem e da referência
patch_img = []
patch_ref = [] 

# Altura e Largura percorrendo a imagem com o stride
# No caso, onde não há sobreposição, o stride horizontal será igual 
# a largura da imagem o stride vertical será igual a altura da imagem
hstride = xsize
vstride = ysize
h = math.ceil(img_np.shape[0] / vstride)
w = math.ceil(img_np.shape[1] / hstride)

# %%
# Percorre dimensões obtidas com stride
for m in range(0, h):
    for n in range(0, w):
        print('Processing tile ', m, n)
        # Índices relativos à altura e largura percorrendo com stride
        i_h = m*vstride
        i_w = n*hstride
        
        # Adiciona Patch da Imagem e Referência caso ele esteja contido na imagem de entrada
        #print('M %d, N %d, Height start %d finish %d , Width start %d finish %d' % (m, n, i_h , i_h+patch_size, i_w, i_w+patch_size) )
        if ( (i_h + ysize <= img_np.shape[0]) and (i_w + xsize <= img_np.shape[1]) ):
            patch_img.append( subimg_np[i_h : i_h+ysize, i_w : i_w+xsize, :] )
            patch_ref.append( subimg_ref_np[i_h : i_h+ysize, i_w : i_w+xsize] )
     
            
# %% 

# Quantidade de pixels em um tile
pixels_tile = ref_tiles.shape[1]*ref_tiles.shape[2]
perc_corte = 1
pixels_corte = pixels_tile*(perc_corte/100)

np.count_nonzero(ref_tiles == 1)

# Esse array mostra a contagem de pixels de estrada para cada tile
# O objetivo é pegar os tiles com maior contagem de pixels e que sejam maiores que 0
contagem_de_estrada = np.array([np.count_nonzero(ref_tiles[i] == 1) for i in range(ref_tiles.shape[0])])

#indices_estradas = np.array([i for i in range(ref_tiles.shape[0])])

ref_tiles_maior_que_1000 = ref_tiles[contagem_de_estrada>10000]

# Esse array são os índices dos tiles que tem número de pixels de estrada maior que 0
indices_maior_0 = np.where(contagem_de_estrada > 0)[0]

#indices_sorted = 


# %% Pega n tiles com maior número 

n = 3

# 
idxs_n = contagem_de_estrada.argsort()[::-1][:n]


# %%

def filtra_tiles_estradas(img_tiles, ref_tiles, perc_corte):
    '''
    

    Parameters
    ----------
    ref_tiles : TYPE
        DESCRIPTION.
    perc_corte : TYPE
        DESCRIPTION.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    if perc_corte < 0 or perc_corte > 100:
        raise Exception('Percentual de pixels de estrada deve estar entre 0 e 100')
        
    
    # Quantidade de pixels em um tile
    pixels_tile = ref_tiles.shape[1]*ref_tiles.shape[2]

    # Quantidade mínima de pixels de estrada     
    pixels_corte = pixels_tile*(perc_corte/100)
    
    # Esse array mostra a contagem de pixels de estrada para cada tile
    # O objetivo é pegar os tiles que tenham número de pixels de estrada maior que o limiar
    contagem_de_estrada = np.array([np.count_nonzero(ref_tiles[i] == 1) for i in range(ref_tiles.shape[0])])
    
    # Esse array são os índices dos tiles que tem número de pixels de estrada maior que o limiar
    indices_maior = np.where(contagem_de_estrada > pixels_corte)[0]  
    
    # Filtra arrays
    img_tiles_filt = img_tiles[indices_maior]
    ref_tiles_filt = ref_tiles[indices_maior]
    
    # Retorna arrays filtrados e índices dos arrays escolhidos
    return img_tiles_filt, ref_tiles_filt, indices_maior

# %% 

img_tiles_filt, ref_tiles_filt, indices_maior = filtra_tiles_estradas(img_tiles, ref_tiles, 1)



# %%

for numero in indices_maior:
    print(numero)

# %% 

tiles_folder = 'teste_tiles'
new_tiles_folder = 'new_teste_tiles'
indices_filtro = indices_maior


linhas = []
colunas = []

for ind in indices_filtro:
    linha = ind // cols
    coluna = ind % cols
    
    linhas.append(linha)
    colunas.append(coluna)
    
tile_files = [f for f in os.listdir(tiles_folder) if os.path.isfile(os.path.join(tiles_folder, f))]
tile_file0 = tile_files[0] 
filename_wo_ext0 = tile_file0.split('.tif')[0]
tile_line0 =  int(filename_wo_ext0.split('_')[1])
tile_col0 = int(filename_wo_ext0.split('_')[2])


shutil.copy(os.path.join(tiles_folder, tile_file0), new_tiles_folder)
# %% 

for tile_file in tile_files:
    filename_wo_ext = tile_file.split('.tif')[0] 
    tile_line = int(filename_wo_ext.split('_')[1])
    tile_col = int(filename_wo_ext.split('_')[2])
    
    for i in range(len(linhas)):
        if tile_line == linhas[i] and tile_col == colunas[i]:
            shutil.copy(os.path.join(tiles_folder, tile_file), new_tiles_folder)
            
        continue




# %% 
def copia_tiles_filtrados(tiles_folder, new_tiles_folder, indices_filtro):
    # Acha linha e coluna do respectivo tile
    linhas = []
    colunas = []
    
    for ind in indices_filtro:
        linha = ind // cols
        coluna = ind % cols
        
        linhas.append(linha)
        colunas.append(coluna)
    
    # Lista com os nomes de arquivos dos tiles na pasta de entrada
    tile_files = [f for f in os.listdir(tiles_folder) if os.path.isfile(os.path.join(tiles_folder, f))]
    
    # Copia arquivos
    for tile_file in tile_files:
        filename_wo_ext = tile_file.split('.tif')[0] 
        tile_line = int(filename_wo_ext.split('_')[1])
        tile_col = int(filename_wo_ext.split('_')[2])
        
        for i in range(len(linhas)):
            if tile_line == linhas[i] and tile_col == colunas[i]:
                shutil.copy(os.path.join(tiles_folder, tile_file), new_tiles_folder)
                
            continue
    

# %%         
        
        