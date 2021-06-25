"""
predict.py
Python file to do automatic liver segmentation on a qMRI dataset. 

@author: dtank
"""

# Imports
from medicaltorch import models as mt_models
from medicaltorch import transforms as mt_transforms
from medicaltorch import datasets as mt_datasets
import nibabel as nib
import argparse 
import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from time import sleep
import os
import sys
sys.path.append('.')
sys.path.append('../src/utils')
from transforms import *
from datasets import *
from models import *
import warnings
warnings.filterwarnings('ignore')

def get_model():
    '''
    Function that loads and returns a prediction model.
    '''
    
    PATH = '../model/model-fold-0.pth'
    model = mt_models.Unet()
    model.load_state_dict(torch.load(PATH))
    
    return model

def data_preprocessing(filename_pair):
    '''
    Returns a dataloader that can be used for predictions. 

    Parameters
    ----------
    filename_pair : a filename pair (slice, mask)

    Returns
    -------
    pred_dataloader : pytorch dataloader

    '''
    pred_transforms = torchvision.transforms.Compose([
                                                      HistogramClipping(),
                                                      RangeNorm(),
                                                      mt_transforms.ToTensor()
                                                      ])

    pred_dataset = MRI2DSegmentationDataset([filename_pair],
                                                   transform=pred_transforms)
    
    pred_dataloader = DataLoader(pred_dataset, batch_size=1,collate_fn=mt_datasets.mt_collate)
    
    return pred_dataloader
    
def get_predictions(pred_dataloader, model):
    """
    Returns model predictions of images in the dataloader.

    Parameters
    ----------
    pred_dataloader : pytorch dataloader
    model : a pytorch (pre-trained) model

    Returns
    -------
    preds : model predictions (list)

    """
    preds = []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(pred_dataloader)):
            sleep(0.01)
            images = batch['input']
            
            model.eval()
            pred = model(images)
            preds.append(np.reshape(pred, [256, 256, 1]))
    return preds

def make_nifti(preds, data_path, save_path):
    '''
    Saves the model predictions as a nifti file.

    Parameters
    ----------
    preds : model predictions
    data_path : nifti input file
    save_path : path where to save the predicted nifti file

    '''
    prediction_stack = np.stack(preds, axis=2)
    prediction_stack = np.array(prediction_stack, dtype=np.int16)
    
    img = nib.load(data_path)
    
    # The affine transformation is deduced from the original nifti image
    pred = nib.Nifti1Image(prediction_stack, img.affine)
    nib.save(pred, os.path.join(save_path+'predictedliver.nii.gz'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='path to a nifti file')
    parser.add_argument('--save_path', help='path where the predictions will be saved')
    
    args = parser.parse_args()
    
    # Get filename from argument parser
    filename_pair = [args.data_path, None]
    
    # Get dataloader with preprocessed images
    pred_dataloader = data_preprocessing(filename_pair)
    
    # Get predictions from trained model 
    predictions = get_predictions(pred_dataloader, get_model())
    
    # Save predictions as nifti file
    make_nifti(predictions, args.data_path, args.save_path)
    
    print('Done')
    