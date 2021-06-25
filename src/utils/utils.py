"""
utils.py

Helper functions that can be used for data analysis and training.

@author: dtank
"""

# Imports
import os
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import yaml

# Functions
def make_filename_pairs(dirlist, liver_path='/Liver2.nii.gz'): # default mask2
    '''
    Makes filename pairs (image_file, mask). Default mask is the whole liver mask, 
    you can change this for 'Liver.nii.gz' to choose the mask excluding the veins. 
    '''
    filename_pairs = []

    for fold in dirlist:
        img_path = fold

        dirs,file=os.path.split(fold)
        mask_path = dirs+liver_path 
        
        try:
            nib.load(mask_path)
            filename_pairs.append((img_path, mask_path))
        except:
            print('No mask for image', img_path)
            pass
        
    return filename_pairs

def slice_filtering(slice_pair):
    ''' This slice filtering technique disregards slices that do not contain
    a mask '''
    if len(np.unique(slice_pair['gt'])) == 1:
        return False
    else:
        return True
    
def load_config(config_path):
    ''' Loads the configuration file'''
    with open(config_path) as file:
        config = yaml.load(file)
    return config

def vis_pair(input_slice, gt_slice):
    ''' Visualizes a filename pair (image and mask)'''
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,6))
    input_slice = ndimage.rotate(input_slice, 90)
    ax1.imshow(input_slice, cmap='gray')
    ax1.set_title('Image')
    ax1.axis('off')
    
    gt_slice = ndimage.rotate(gt_slice, 90)
    ax2.imshow(gt_slice, cmap='gray')
    ax2.set_title('Mask')
    ax2.axis('off')

def vis_batch(dataloader):
    ''' Visualizes a batch from a pytorch dataloader '''
    batch = next(iter(dataloader))

    for i in range(len(batch['input'])):

        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,6))

        input_slice = batch['input'][i].squeeze(0)
        input_slice = ndimage.rotate(input_slice, 90)
        gt_slice = batch['gt'][i].squeeze(0)
        gt_slice = ndimage.rotate(gt_slice, 90)
        
        ax1.imshow(input_slice, cmap='gray')
        ax1.set_title('Image')
        ax1.axis('off')

        ax2.imshow(gt_slice, cmap='gray')
        ax2.set_title('Mask')
        ax2.axis('off')

    
def sub_filename_pairs(filename_pairs, train_index, test_index):
    ''' Makes filename pairs for '''
    train = []
    test = []
    for index in train_index:
        train.append(filename_pairs[index])
    for index in test_index:
        test.append(filename_pairs[index])
    return train, test

def reset_weights(m):
    ''' Resets model weights, used for k-fold cross validation'''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
