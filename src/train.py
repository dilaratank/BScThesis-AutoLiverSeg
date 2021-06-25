#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train.py
File to train four different networks on qMRI data.

@author: dtank
"""

# Imports
import argparse
from sklearn.model_selection import KFold
import sys  
sys.path.append('.')
sys.path.append('../src/utils')
from datasets import *
from transforms import *
from models import *
from utils import *
from medicaltorch import transforms as mt_transforms
from medicaltorch import datasets as mt_datasets
from medicaltorch import models as mt_models
from medicaltorch import losses as mt_losses
# import yaml
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import IO as io
import warnings
warnings.filterwarnings('ignore')

def training_loop(network, train_dataloader, val_dataloader, num_epochs, device, optimizer, fold):
    """
    This function takes care of the training loop for automatic liver segmentation.
    The training loss is calculated here and stored. The validation is also performed 
    in this loop to keep track of early stopping.
    """
    
    early_stopping_counter = 0
    best_loss = 0
    for epoch in range (0, num_epochs):
        
        print(f'Starting epoch {epoch+1}')
        
        # Set current loss value
        current_loss = 0
        
        # Iterate over dataloader for training data
        for i, batch in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            
            # Get inputs
            images, labels = batch["input"], batch["gt"]
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad() 
            
            # Perform forward pass
            network.train()
            preds = network(images) 
            
            # Compute loss
            if config['train_params']['loss'] == 'dice_loss':
                loss = mt_losses.dice_loss(preds, labels)
            elif config['train_params']['loss'] == 'BCE':
                lossf = torch.nn.BCELoss()
                loss = lossf(preds, labels)
            # TODO: you can add other losses here
            
            # Perform backward pass
            loss.backward() 
            
            # Perform optimization 
            optimizer.step() 
            
            current_loss += loss.item()
            
                
        # Process complete, print statistics, validate and save model
        print(f'Training process epoch {epoch} finished')
        print('Current loss:', current_loss/len(train_dataloader))
        tb.add_scalar(f'Train Loss Fold {fold}', current_loss/len(train_dataloader), global_step=epoch)
       
        # Validation
        print(f'Validating epoch {epoch}')
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, position=0, leave=True):
    
                # Get inputs
                images, labels = batch['input'], batch['gt']
                images, labels = images.to(device), labels.to(device)
    
                # Generate outputs 
                network.eval()
                preds = network(images)
                
                loss = mt_losses.dice_loss(preds, labels)
                val_loss += loss.item()
        
        mean_val_loss = val_loss/len(val_dataloader)
        
        print('mean validation loss', mean_val_loss)
        print('best loss', best_loss)
        
        if epoch % 5 == 0:
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                save_path = f'./model-{args.arch}-fold-{fold}.pth'
                torch.save(network.state_dict(), save_path)
                
                # Uncomment to save for transfer learning
                # state = {'epoch': epoch, 'state_dict': network.state_dict(), 'optimizer': optimizer.state_dict()}
                # torch.save(state, f'./model-state-fold-{fold}.pth')
                
            else:
                early_stopping_counter += 1
            
        tb.add_scalar(f'Validation Loss Fold {fold}', mean_val_loss, global_step=epoch)
        
        if early_stopping_counter == 25:
            print('Early stopping')
            break
        
        print(f'Early stopping counter: {early_stopping_counter}')
            
def testing_loop(network, test_dataloader, results, device, fold):
    """
    This function takes care of the testing loop. The final results are stored
    here.
    """
    
    print('Start testing')
    
    # Total score
    total_score, total = 0, 0

    # Evaluation for current fold
    with torch.no_grad():
        for batch in tqdm(test_dataloader, position=0, leave=True):

            # Get inputs
            images, labels = batch['input'], batch['gt']
            images, labels = images.to(device), labels.to(device)

            # Generate outpuds 
            network.eval()
            pred = network(images)

            # Calculate dice coefficient 
            dsc = -1 * mt_losses.dice_loss(pred, labels)
            print('dice score:', dsc.item())
            total_score += dsc.item()
            total += 1

        # Print dice coefficient score 
        print(f'Dice score for fold {fold}:', total_score/total)
        tb.add_scalar(f'Dice Score Test Fold {fold}', total_score/total)
        print('total score:', total_score)
        print('total preds:', total)
        print('------------------------------')
        results[fold] = total_score/total
    
    return results

def train(config, args):
    """
    This function takes care of the training for automatic liver segmentation, 
    The first part consists of defining the data, datasets, dataloaders, model,
    optimizers, etc.
    The second part consists of the training- and validation loop.
    The third part consists of the testing loop.
    """
    
    # Get data filenames
    dirlist = io.main(config['dataset']['data_directory'])
    filename_pairs = make_filename_pairs(dirlist)
    # Uncomment this to train on the mask without the veins
    # filename_pairs = make_filename_pairs(dirlist, liver_path='/Liver.nii.gz')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device, torch.cuda.current_device())
    
    
    # Define preprocessing transforms
    preprocessing_transforms = torchvision.transforms.Compose([
                                                        HistogramClipping(),
                                                        RangeNorm(),
                                                        mt_transforms.ToTensor()])
    # Define transformations
    train_transforms = torchvision.transforms.Compose([
            mt_transforms.RandomRotation(config['train_dataloader']['data_aug']['rot_degree']), # random rotation
            mt_transforms.RandomAffine(0, translate=config['train_dataloader']['data_aug']['transl_range']), # shift
            mt_transforms.RandomAffine(0, shear=config['train_dataloader']['data_aug']['shear_range']), # shear
            HistogramClipping(),
            RangeNorm(),
            mt_transforms.ToTensor(),
        ])
    
    
    # Create a dict to store fold results
    results = {}
    
    # Set fixed random number seed so that pseudo random number  
    # initializers will be initialized using the same initialization token.    
    # Define cross validatior
    kfold = KFold(n_splits=config['model_params']['k_folds'], 
                  shuffle=True, random_state=config['model_params']['random_state'])
    
    print('------------------------------')
    
    # Start training per fold
    for fold, (train_index, test_index) in enumerate(kfold.split(filename_pairs)):
        
        
        print(f'FOLD {fold}')
        print('------------------------------')
        
        # Get train and test filenames
        train_files, test_files = sub_filename_pairs(filename_pairs, train_index, test_index)
        
        # Get train and validation filenames
        train_files, val_files = train_test_split(train_files, test_size=0.1, 
                                                  random_state=config['model_params']['random_state'])
        
        # Define datasets
        print('start preparing datasets')
        
        # Unet and Segnet datasets
        if args.arch == 'unet' or args.arch == 'segnet':
            train_dataset = MRI2DSegmentationDataset(train_files,
                                                     transform=train_transforms, 
                                                     preprocess=preprocessing_transforms,
                                                     slice_filter_fn=slice_filtering,
                                                     p=0.8)
            
            val_dataset = MRI2DSegmentationDataset(val_files, 
                                                   preprocess=preprocessing_transforms,
                                                   slice_filter_fn=slice_filtering)
            
            test_dataset = MRI2DSegmentationDataset(test_files,
                                                    preprocess=preprocessing_transforms,
                                                    slice_filter_fn=slice_filtering)
            
        # Multichannel Unet datasets
        if args.arch == 'multichannel-unet':
            # Note that transformations for this type of unet have not been implemented and cannot be used
            
            train_dataset = MultiChannelMRI2DSegmentationDataset(train_files,
                                                                  transform=torchvision.transforms.Compose([mt_transforms.ToTensor()]),
                                                                  slice_filter_fn=slice_filtering)
        
            val_dataset = MultiChannelMRI2DSegmentationDataset(val_files, 
                                                                transform=torchvision.transforms.Compose([mt_transforms.ToTensor()]),
                                                                slice_filter_fn=slice_filtering)
            
            test_dataset = MultiChannelMRI2DSegmentationDataset(test_files,
                                                                 transform=torchvision.transforms.Compose([mt_transforms.ToTensor()]),
                                                                 slice_filter_fn=slice_filtering)
        # Average Image Unet datasets
        if args.arch == 'avgimg-unet':
            train_dataset = AvgImgMRI2DSegmentationDataset(train_files,
                                                           transform=train_transforms, 
                                                           preprocess=preprocessing_transforms,
                                                           slice_filter_fn=slice_filtering,
                                                           p=0.8)
            
            val_dataset = AvgImgMRI2DSegmentationDataset(val_files, 
                                                         preprocess=preprocessing_transforms,
                                                         slice_filter_fn=slice_filtering)
            
            test_dataset = AvgImgMRI2DSegmentationDataset(test_files,
                                                          preprocess=preprocessing_transforms,
                                                          slice_filter_fn=slice_filtering)
            

        # Define dataloaders
        print('start preparing dataloaders')
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=config['train_dataloader']['batch_size'],
                                      shuffle=config['train_dataloader']['shuffle'],
                                      collate_fn=mt_datasets.mt_collate)
        
        val_dataloader = DataLoader(val_dataset, 
                                    batch_size=config['test_dataloader']['batch_size'],
                                    collate_fn=mt_datasets.mt_collate)
        
        test_dataloader = DataLoader(test_dataset, 
                                     batch_size=config['test_dataloader']['batch_size'],
                                     collate_fn=mt_datasets.mt_collate)
        
        # Initialize the neural network
        if args.arch == 'unet' or args.arch == 'avgimg-unet':
            network = mt_models.Unet()
        if args.arch == 'multichannel-unet':
            network = MultiChannelUnet()
        if args.arch == 'segnet':
            network = SegNet(1)
        
            
        # Weights are reset so that each fold begins with uninitialized weights
        network.apply(reset_weights)
        network.to(device)
        
        # Initialize optimizer
        if config['train_params']['optimizer'] == 'adam':
            optimizer = optim.Adam(network.parameters(), lr=config['train_params']['lr'])
        # TODO: here you can add other optimizers         
        
        # Train
        print('start training loop')
        training_loop(network, train_dataloader, val_dataloader,
                      config['train_params']['num_epochs'], 
                      device, optimizer, fold)

        # Get best model        
        PATH = f'./model-{args.arch}-fold-{fold}.pth'
        
        if args.arch == 'unet' or args.arch == 'avgimg-unet':
            testnetwork = mt_models.Unet()
            
        if args.arch == 'multichannel-unet':
            testnetwork = MultiChannelUnet()
            
        if args.arch == 'segnet':
            testnetwork = SegNet(1)
        
        testnetwork.load_state_dict(torch.load(PATH))
        testnetwork.to(device)
        
        # Test
        results = testing_loop(testnetwork, test_dataloader, 
                               results, device, fold)
        
        # Uncomment to train on 1 fold only
        break 
        
    # Print fold results
    k_folds = config['model_params']['k_folds']
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')  
    print('------------------------------')
    sum_res = 0
    for key, value in results.items():
        print(f'Fold {key}: {value}')
        # tb.add_text(f'Fold {key}: {value}')
        sum_res += value
    print('Average:', sum_res/len(results.items()))

if __name__ == "__main__":
    
    CONFIG_PATH = '/home/dtank/scratch/AutoLiverSeg/src/config.yaml'
    config = load_config(CONFIG_PATH)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default='unet', type=str,
                        help="Network architectures, possible value: 'unet', 'multichannel-unet', 'avgimg-unet', 'segnet'" )
    parser.add_argument('--name', type=str, help='The name for tensorboard')
    args = parser.parse_args()
    
    # Tensorboard logs setup
    comment = args.name
    print(comment)
    tb = SummaryWriter(comment=comment)
    
    # Start training process
    train(config, args)
    
    tb.close()