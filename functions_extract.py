# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 13:21:49 2024

@author: Marcel
"""
from osgeo import gdal, gdal_array
import numpy as np
import math, os, shutil
from sklearn.preprocessing import MinMaxScaler


def load_tiff_image(image):
    print(image)
    gdal_header = gdal.Open(image)
    img_gdal = gdal_header.ReadAsArray()
    img = np.transpose(img_gdal, (1,2,0)) # Transpõe imagem para as bandas ficarem
                                          # na última dimensão
    print(img.shape)
    return img


def load_tiff_image_reference(image):
    print(image)
    gdal_header = gdal.Open(image)
    img_gdal = gdal_header.ReadAsArray()
    #img = np.expand_dims(img_gdal, 2)
    img = img_gdal
    print(img.shape)
    return img

def normalization(image):
    # Primeiro remocela a imagem planificando as linhas e colunas para fazer normalização
    # Depois remodela para formato original
    image_reshaped = image.reshape(image.shape[0]*image.shape[1], image.shape[2])
    scaler = MinMaxScaler(feature_range=(0,1))
    image_normalized_ = scaler.fit_transform(image_reshaped)
    image_normalized = image_normalized_.reshape(image.shape[0], image.shape[1], image.shape[2])
    
    return image_normalized

# Função para extrair os patches
def extract_patches(image, reference, patch_size, stride, border_patches=False):
    '''
    Function: extract_patches
    -------------------------
    Extract patches from the original and reference image
    
    Input parameters:
      image      = array containing the original image (h,w,c)
      reference  = array containing the reference image (h,w)
      patch_size = patch size (scalar). The shape of the patch is square.
      stride     = displacement to be applied.
      border_patches = include patches overlaping image borders (at most only one for each line or column and only when necessary to complete 
                                                                 the image)
    
    Returns: 
      A, B = List containing the patches for the input image (A) and respective reference (B).
    '''
    # Listas com os patches da imagem e da referência
    patch_img = []
    patch_ref = []
    
    # Quantidade de canais da imagem (canais na última dimensão - channels last)
    image_channels = image.shape[-1]
    
    # Altura e Largura percorrendo a imagem com o stride
    h = math.ceil(image.shape[0] / stride)
    w = math.ceil(reference.shape[1] / stride)
    
    
    # Acha primeiro valor de m e n que completa ou transborda a imagem 
    # Imagem terá math.floor(image.shape[0] / stride) patches na vertical e 
    # math.floor(image.shape[1] / stride) patches na horizontal, caso border_pathes=False.
    # Imagem terá firstm_out+1 patches na vertical e 
    # firstn_out+1 patches na horizontal, caso border_pathes=True.
    for m in range(0, h):
        i_h = m*stride
        if i_h + patch_size == image.shape[0]:
            break
        if i_h + patch_size > image.shape[0]: 
            break
    
    firstm_out = m
    
    for n in range(0, w):
        i_w = n*stride
        if i_w + patch_size == image.shape[1]: 
            break
        if i_w + patch_size > image.shape[1]: 
            break
    
    firstn_out = n
    
    
    # Percorre dimensões obtidas com stride
    for m in range(0, h):
        for n in range(0, w):
            # Índices relativos à altura e largura percorrendo com stride
            i_h = m*stride
            i_w = n*stride
            
            # Adiciona Patch da Imagem e Referência caso ele esteja contido na imagem de entrada
            #print('M %d, N %d, Height start %d finish %d , Width start %d finish %d' % (m, n, i_h , i_h+patch_size, i_w, i_w+patch_size) )
            if ( (i_h + patch_size <= image.shape[0]) and (i_w + patch_size <= image.shape[1]) ):
                patch_img.append( image[i_h : i_h+patch_size, i_w : i_w+patch_size, :] )
                patch_ref.append( reference[i_h : i_h+patch_size, i_w : i_w+patch_size] )
                
            # Trata bordas no caso que o patch esteja parcialmente contido na imagem de entrada (parte do patch está fora da borda)
            # Preenche o que sai da borda com 0s
            elif border_patches:
                # Inicia Patches de Borda (imagem e referência) com 0s 
                border_patch_img = np.zeros((patch_size, patch_size, image_channels))
                border_patch_ref = np.zeros((patch_size, patch_size))
                
                # Se patch ultrapassa altura da imagem,
                # border_mmax é o que falta desde o início do patch até a borda inferior da imagem
                if (i_h + patch_size > image.shape[0]):
                    border_mmax = image.shape[0] - i_h
                # Caso contrário mantém-se o tamanho do patch   
                else:
                    border_mmax = patch_size
                
                # Se patch ultrapassa largura da imagem,
                # border_nmax é o que falta desde o início do patch até a borda direita da imagem     
                if (i_w + patch_size > image.shape[1]):
                    border_nmax = image.shape[1] - i_w
                else:
                    border_nmax = patch_size
                    
                    
                # Preenche patches
                border_patch_img[0:border_mmax, 0:border_nmax, :] = image[i_h : i_h+border_mmax, i_w : i_w+border_nmax, :]
                border_patch_ref[0:border_mmax, 0:border_nmax] = reference[i_h : i_h+border_mmax, i_w : i_w+border_nmax]
                
                
                # Adiciona patches à lista somente se m e n forem os menores valores a transbordar a imagem
                if m <= firstm_out and n <= firstn_out:
                    patch_img.append( border_patch_img )
                    patch_ref.append( border_patch_ref )
      
        
    # Retorna os arrays de patches da imagem e da referência        
    return np.array(patch_img), np.array(patch_ref)


           
# Função que faz a extração de tiles em uma imagem
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

# Função que retorna os tiles igual ou acima de um certo percentual de pixels de estrada
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


# Função que copia os tiles filtrados (por filtra_tiles_estradas) desde uma pasta para outra
# cols é o número de tiles na horizontal, indices_filtro pode ser a variável indices_maior retornada por
# por filtra_tiles_estradas
def copia_tiles_filtrados(tiles_folder, new_tiles_folder, indices_filtro, cols):
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
        
# A partir de dicionário com endereço de tile:máscara, é feita extração dos patches no tile        
def extract_patches_from_tiles(imgs_labels_dict, patch_size, patch_stride, border_patches=False):
    tile_items = list(imgs_labels_dict.items())
    
    x_patches, y_patches, len_tiles, shape_tiles = [], [], [], []
    
    for i in range(len(tile_items)):
        tile_item = tile_items[i]
        
        img_tile = load_tiff_image(tile_item[0]) 
        #img_tile = normalization(img_tile)
        label_tile = load_tiff_image_reference(tile_item[1])
        
        print(f'tile_items img {i} =' , tile_item[0])
        print(f'tile_items label {i} =' , tile_item[1])
        
        # Add to list 
        x_patches_new, y_patches_new = extract_patches(img_tile, label_tile, patch_size, patch_stride, border_patches=border_patches)
        
        x_patches.append(x_patches_new)
        y_patches.append(y_patches_new)
        len_tiles.append(len(x_patches_new))
        shape_tiles.append(img_tile.shape[:2])
           

    # Concatenate patches
    x_patches = np.concatenate(x_patches, axis=0)
    y_patches = np.concatenate(y_patches, axis=0)

    # Transform y_patches, shape (N, H, W) into shape (N, H, W, C). Necessary for data agumentation.
    y_patches = np.expand_dims(y_patches, 3)
            
    return x_patches, y_patches, len_tiles, shape_tiles


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

