#!/bin/bash

"""
This is a script derived from the tutorial for solaris
found here:
https://solaris.readthedocs.io/en/latest/tutorials/notebooks/map_vehicles_cowc.html
This is the precursor to the demo on the utilization
of federated learning on GIS data and satellite imagery
"""

import solaris as sol
import os
import glob
import gdal
from tqdm import tqdm
import cv2
import shutil
import pandas as pd
import numpy as np
from skimage.morphology import square, dilation
from matplotlib import pyplot as plt
from solaris.eval.iou import calculate_iou

# Identify directories for data once organized
root= "/home/sarah/solaris_cowc/cowc/datasets/ground_truth_sets"  ##cowc ground_truth_sets location after download
masks_out= "/home/sarah/solaris_cowc/cowc/masks" ##output location for your masks for training
images_out= "/home/sarah/solaris_cowc/cowc/tiles" ##output location for your tiled images for testing
masks_test_out= "/home/sarah/solaris_cowc/cowc/masks_test" ##output location for your masks for testing
images_test_out= "/home/sarah/solaris_cowc/cowc/tiles_test" ##output location for your tiled images for testingimport geopandas as gpd

# Initialize a tiling function
def geo_tile(untiled_image_dir, tiles_out_dir, tile_size=544,
             overlap=0.2, search=".png",Output_Channels=[1,2,3]):
    """Function to tile a set of images into smaller square chunks with embedded georeferencing info
    allowing an end user to specify the size of the tile, the overlap of each tile, and when to discard
    a tile if it contains blank data.
    Arguments
    ---------
    untiled_image_dir : str
        Directory containing full or partial image strips that are untiled.
        Imagery must be georeferenced.
    tiles_out_dir : str
        Output directory for tiled imagery.
    tile_size : int
        Extent of each tile in both X and Y directions in units of pixels.
        Defaults to ``544`` .
    overlap : float
        The amount of overlap of each tile in float format.  Should range between 0 and <1.
        Defaults to ``0.2`` .
    search : str
        A string with a wildcard to search for files by type
        Defaults to ".png"
    Output_Channels : list
        A list of the number of channels to output, 1 indexed.
        Defaults to ``[1,2,3]`` .
    Returns
    -------
    Tiled imagery directly output to the tiles_out_dir
    """
    if not os.path.exists(tiles_out_dir):
        os.makedirs(tiles_out_dir)

    os.chdir(untiled_image_dir)
    search2 = "*" + search
    images = glob.glob(search2)
    tile_size = int(tile_size)

    for stackclip in images:
        print(stackclip)
        interp = gdal.Open(os.path.abspath(stackclip))
        width = int(interp.RasterXSize)
        height = int(interp.RasterYSize)
        count = 0
        for i in range(0, width, int(tile_size * (1 - overlap))):
            for j in range(0, height, int(tile_size * (1 - overlap))):
                Chip = [i, j, tile_size, tile_size]
                count += 1
                Tileout = tiles_out_dir + "/" + \
                    stackclip.split(search)[0] + "_tile_" + str(count) + ".tif"
                output = gdal.Translate(Tileout, stackclip, srcWin=Chip, bandList=Output_Channels)
                del output
    print("Done")

# Organize the data
os.chdir(root)
dirs=glob.glob("*/")
for directory in dirs:
    os.chdir(directory)
    if not os.path.exists("Images"):
        os.makedirs("Images")
        os.makedirs("Masks")
        os.makedirs("Extras")
    xcfs=glob.glob("*.xcf")
    txts=glob.glob("*.txt")
    os.chdir("Images")
    negatives=glob.glob("*Negatives.png")
    masks=glob.glob("*Annotated_Cars.png")
    for xcf in xcfs:
        shutil.move(xcf,os.path.join(root,directory,"Extras",xcf))
    for txt in txts:
        shutil.move(txt,os.path.join(root,directory,"Extras",txt))
    for negative in negatives:
        shutil.move(negative,os.path.join(root,directory,"Extras",negative))
    for mask in masks:
        shutil.move(mask,os.path.join(root,directory,"Masks",mask))
    images=glob.glob("*.png")
    for image in images:
        shutil.move(image,os.path.join(root,directory,"Images",image))
    os.chdir(root)

# Tiling, masking and converting to GeoTiffs
for directory in dirs:
    if directory != "Utah_AGRC":
        directory = os.path.join(root,directory,"Masks")
        print(directory)
        geo_tile(directory, masks_out, tile_size=512, overlap=0.1,search="*.png",Output_Channels=[1])
    else:
        directory = os.path.join(root,directory,"Masks")
        print(directory)
        geo_tile(directory, masks_out, tile_size=512, overlap=0,search="*.png",Output_Channels=[1]) #No overlap for testing.

for directory in dirs:
    if directory != "Utah_AGRC":
        directory = os.path.join(root,directory,"Images")
        print(directory)
        geo_tile(directory, images_out, tile_size=512, overlap=0.1,search="*.png",Output_Channels=[1,2,3])
    else:
        directory = os.path.join(root,directory,"Images")
        print(directory)
        geo_tile(directory, images_out, tile_size=512, overlap=0,search="*.png",Output_Channels=[1,2,3])

# Dilate the masks to increase size of labels
# Using simple morphological dilation filter

driver = gdal.GetDriverByName("GTiff")
os.chdir(masks_out)
images=glob.glob("*.tif")
for image in tqdm(images):
    band=gdal.Open(image)
    band = band.ReadAsArray()
    band=dilation(band, square(9))
    im_out = driver.Create(image,band.shape[1],band.shape[0],1,gdal.GDT_Byte)
    im_out.GetRasterBand(1).WriteArray(band)
    del im_out

# Calculate the z-scores
M1=[]
M2=[]
M3=[]
S1=[]
S2=[]
S3=[]
driver = gdal.GetDriverByName("GTiff")
os.chdir(images_out)
images=glob.glob("*.tif")
for image in images:
    band=gdal.Open(image).ReadAsArray()
    M1.append(np.mean(band[0,:,:]))
    M2.append(np.mean(band[1,:,:]))
    M3.append(np.mean(band[2,:,:]))
    S1.append(np.std(band[0,:,:]))
    S2.append(np.std(band[1,:,:]))
    S3.append(np.std(band[2,:,:]))

print("Save these numbers for your solaris.yml file for training and z-scoring (normalizing) your imagery")
print(np.mean(M1)/255)
print(np.mean(M2)/255)
print(np.mean(M3)/255)
print(np.mean(S1)/255)
print(np.mean(S2)/255)
print(np.mean(S3)/255)

# Hold out a city for testing
if not os.path.exists(images_test_out):
        os.makedirs(images_test_out)
os.chdir(images_out)
images = glob.glob("12TVL*")
for image in tqdm(images):
    output = os.path.join(images_test_out,image)
    shutil.move(image, output)

if not os.path.exists(masks_test_out):
        os.makedirs(masks_test_out)
os.chdir(masks_out)
images = glob.glob("12TVL*")
for image in tqdm(images):
    output = os.path.join(masks_test_out,image)
    shutil.move(image, output)

# Create a csv for images and masks for training and testing
data = []
images = []
image_folder=images_out
label_folder=masks_out
os.chdir(label_folder)
labels=glob.glob("*.tif")
for x in labels:
    z = x.split('_Annotated_Cars')[0] + x.split('_Annotated_Cars')[1]
    os.chdir(image_folder)
    image=glob.glob(z)
    if len(image) != 1:
        os.chdir(label_folder)
        os.remove(x)
    else:
        images.append(image[0])

for image, label in zip(images,labels):
    image = os.path.join(image_folder,image)
    label = os.path.join(label_folder,label)
    data.append((image, label))

df = pd.DataFrame(data, columns=['image', 'label'])
df.to_csv(os.path.join(root,"train_data_cowc2.csv"))

data = []
images = []
image_folder=images_test_out
label_folder=masks_test_out
os.chdir(label_folder)
labels=glob.glob("*.tif")
for x in labels:
    z = x.split('_Annotated_Cars')[0] + x.split('_Annotated_Cars')[1]
    os.chdir(image_folder)
    image=glob.glob(z)
    if len(image) != 1:
        os.chdir(label_folder)
        os.remove(x)
    else:
        images.append(image[0])

for image, label in zip(images,labels):
    image = os.path.join(image_folder,image)
    label = os.path.join(label_folder,label)
    data.append((image, label))

df = pd.DataFrame(data, columns=['image', 'label'])
df.to_csv(os.path.join(root,"test_data_cowc2.csv"))

print('Now take those numbers from above and stick them in the yaml')
