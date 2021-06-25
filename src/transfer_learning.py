#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transfer_learning.py

Training process to use with transfer learning. You will need pre-trained
weights. 

@author: dtank
"""

# imports
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
        
        # print epoch
        print(f'Starting epoch {epoch+1}')
        
        # set current loss value
        current_loss = 0
        
        # iterate over dataloader for training data
        for i, batch in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            
            # get inputs
            images, labels = batch["input"], batch["gt"]
            images, labels = images.to(device), labels.to(device)

            # zero the gradients
            optimizer.zero_grad() 
            
            # perform forward pass
            network.train()
            preds = network(images) 
            
            # compute loss
            if config['train_params']['loss'] == 'dice_loss':
                loss = mt_losses.dice_loss(preds, labels)
                # tb.add_scalar('loss', loss.item())
            elif config['train_params']['loss'] == 'BCE':
                lossf = torch.nn.BCELoss()
                loss = lossf(preds, labels)
            
            # perform backward pass
            loss.backward() 
            
            # perform optimization 
            optimizer.step() 
            
            current_loss += loss.item()
            
                
        # process complete, print statistics, validate and save model
        print(f'Training process epoch {epoch} finished')
        print('Current loss:', current_loss/len(train_dataloader))
        tb.add_scalar(f'Train Loss Fold {fold}', current_loss/len(train_dataloader), global_step=epoch)
        # tb.flush()
       
        # validation
        print(f'Validating epoch {epoch}')
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, position=0, leave=True):
    
                # get inputs
                images, labels = batch['input'], batch['gt']
                images, labels = images.to(device), labels.to(device)
    
                # generate outpuds 
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
                save_path = f'./model-fold-{fold}.pth'
                torch.save(network.state_dict(), save_path)      
            else:
                early_stopping_counter += 1
            
        tb.add_scalar(f'Validation Loss Fold {fold}', mean_val_loss, global_step=epoch)
        
        if early_stopping_counter == 25:
            print('Early stopping, Model saved')
            break
        
        print(f'Early stopping counter: {early_stopping_counter}')
            
def testing_loop(network, test_dataloader, results, device, fold):
    """
    This function takes care of the testing loop. The final results are stored
    here.
    """
    
    print('Start testing')
    
    # total score
    total_score, total = 0, 0

    # evaluation for current fold
    with torch.no_grad():
        for batch in tqdm(test_dataloader, position=0, leave=True):

            # get inputs
            images, labels = batch['input'], batch['gt']
            images, labels = images.to(device), labels.to(device)

            # generate outpuds 
            network.eval()
            pred = network(images)

            # calculate dice coefficient 
            dsc = -1 * mt_losses.dice_loss(pred, labels)
            print('dice score:', dsc.item())
            total_score += dsc.item()
            total += 1

        # print dice coefficient score 
        print(f'Dice score for fold {fold}:', total_score/total)
        tb.add_scalar(f'Dice Score Test Fold {fold}', total_score/total)
        print('total score:', total_score)
        print('total preds:', total)
        print('------------------------------')
        results[fold] = total_score/total
    
    return results

def train(config):
    # get data
    dirlist = io.main(config['dataset']['data_directory'])
    filename_pairs = make_filename_pairs(dirlist, liver_path='/Liver.nii.gz')
    
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    
    # define transformations
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
    
    # fold results
    results = {}
    
    # set fixed random number seed so that pseudo random number  
    # initializers will be initialized using the same initialization token.    
    # define cross validatior
    kfold = KFold(n_splits=config['model_params']['k_folds'], 
                  shuffle=True, random_state=config['model_params']['random_state'])
    
    # start print
    print('------------------------------')
    
    for fold, (train_index, test_index) in enumerate(kfold.split(filename_pairs)):
        print(f'FOLD {fold}')
        print('------------------------------')
        
        # get train and test filenames
        train_files, test_files = sub_filename_pairs(filename_pairs, train_index, test_index)
        
        # get train and validation filenames
        train_files, val_files = train_test_split(train_files, test_size=0.1, 
                                                  random_state=config['model_params']['random_state'])
        
        # define datasets
        print('start preparing datasets')
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
        
        # define dataloaders
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
        
        # init the neural network
        if config['model_params']['architecture'] == 'unet':
            network = mt_models.Unet()
        elif config['model_params']['architecture'] == 'segnet':
            pass #TODO: add segnet model
        # network.apply(reset_weights)
        network.to(device)
        
        # initialize optimizer
        if config['train_params']['optimizer'] == 'adam':
            optimizer = optim.Adam(network.parameters(), lr=0.0001) # lower optimizer
            
        # Handles chekcpoint loading
        print('loading checkpoint')
        filename = f'./model-state-fold-{fold}.pth'
        checkpoint = torch.load(filename)
        network.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('checkpoint loaded')
        
        # train
        print('start training loop')
        training_loop(network, train_dataloader, val_dataloader,
                      config['train_params']['num_epochs'], 
                      device, optimizer, fold)

        # Get best model        
        PATH = f'./model-fold-{fold}.pth'
        testnetwork = mt_models.Unet()
        testnetwork.load_state_dict(torch.load(PATH))
        testnetwork.to(device)
        
        # Test
        results = testing_loop(testnetwork, test_dataloader, 
                               results, device, fold)
        
        # Uncomment to train on one fold only
        # break 
        
    # Print fold results
    k_folds = config['model_params']['k_folds']
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')  
    print('------------------------------')
    sum_res = 0
    for key, value in results.items():
        print(f'Fold {key}: {value}')
        sum_res += value
    print('Average:', sum_res/len(results.items()))
    
if __name__ == "__main__":
    
    CONFIG_PATH = '/home/dtank/scratch/AutoLiverSeg/src/config.yaml'
    config = load_config(CONFIG_PATH)
    
    # Tensorboard logs setup
    comment = 'Transfer learning with liver1, smaller lr'
    print(comment)
    tb = SummaryWriter(comment=comment)
    
    train(config)
    
    tb.close()