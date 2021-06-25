"""
transforms.py

Transform functions that add to the transforms already present in the 
MedicalTorch package. 

@author: dtank
"""

from medicaltorch import transforms as mt_transforms
import numpy as np
from PIL import Image

### TRANSFORMS ###
class HistogramClipping(mt_transforms.MTTransform):
    ''' Performs histogram clipping. '''
    def __init__(self, min_percentile=1.0, max_percentile=99.0):
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    def __call__(self, sample):
        
        rdict = {}
        input_data = sample['input']
        np_input_data = np.array(input_data)
        
        mask = np.where(np_input_data>0)
        masked = np_input_data[mask].flatten()
        
        percentile1 = np.percentile(masked, self.min_percentile)
        percentile2 = np.percentile(masked, self.max_percentile)
        
        np_input_data[np_input_data <= percentile1] = percentile1
        np_input_data[np_input_data >= percentile2] = percentile2
    
        input_data = Image.fromarray(np_input_data, mode='F')
        rdict['input'] = input_data
        
        sample.update(rdict)
        return sample
    
class RangeNorm(mt_transforms.MTTransform):
    ''' Performs range/intensity normalization. '''
    def __init__(self, minv=0, maxv=255):
        self.min = minv
        self.max = maxv

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']
        np_input_data = np.array(input_data)

        maxv = np.max(np_input_data)

        np_input_data = (np_input_data / maxv) * 255
        
        input_data = Image.fromarray(np_input_data, mode='F')
        rdict['input'] = input_data
        
        sample.update(rdict)
        return sample
