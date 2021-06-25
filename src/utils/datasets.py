"""
datasets.py

Medicaltorch source code, adapted for custom data usage.
The adaptations are documented with the comment 'Changed:' following a 
description. 

@author: dtank
"""

# Imports
from PIL import Image
import numpy as np
from tqdm import tqdm
import nibabel as nib
import torchvision
from torch.utils.data import Dataset

from medicaltorch import datasets as mt_datasets
from medicaltorch import transforms as mt_transforms
from transforms import *



# Adapted source code

### BASELINE ###
class SegmentationPair2D(object):
    """This class is used to build 2D segmentation datasets. It represents
    a pair of of two data volumes (the input data and the ground truth data).
    :param input_filename: the input filename (supported by nibabel).
    :param gt_filename: the ground-truth filename.
    :param cache: if the data should be cached in memory or not.
    :param canonical: canonical reordering of the volume axes.
    """
    def __init__(self, input_filename, gt_filename, cache=True,
                 canonical=False):
        self.input_filename = input_filename
        self.gt_filename = gt_filename
        self.canonical = canonical
        self.cache = cache

        self.input_handle = nib.load(self.input_filename)

        # Unlabeled data (inference time)
        if self.gt_filename is None:
            self.gt_handle = None
        else:
            self.gt_handle = nib.load(self.gt_filename)

        if len(self.input_handle.shape) > 3:
            # Changed: instead of throwing a warning, change the dimension
            # of the input
            self.input_handle = nib.funcs.four_to_three(self.input_handle)[0]

        # Sanity check for dimensions, should be the same
        input_shape, gt_shape = self.get_pair_shapes()

        if self.gt_handle is not None:
            if not np.allclose(input_shape, gt_shape):
                raise RuntimeError('Input and ground truth with different dimensions.')

        if self.canonical:
            self.input_handle = nib.as_closest_canonical(self.input_handle)

            # Unlabeled data
            if self.gt_handle is not None:
                self.gt_handle = nib.as_closest_canonical(self.gt_handle)

    def get_pair_shapes(self):
        """Return the tuple (input, ground truth) representing both the input
        and ground truth shapes."""
        input_shape = self.input_handle.header.get_data_shape()

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_shape = None
        else:
            gt_shape = self.gt_handle.header.get_data_shape()

        return input_shape, gt_shape

    def get_pair_data(self):
        """Return the tuble (input, ground truth) with the data content in
        numpy array."""
        cache_mode = 'fill' if self.cache else 'unchanged'
        input_data = self.input_handle.get_fdata(cache_mode, dtype=np.float32)

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_data = None
        else:
            gt_data = self.gt_handle.get_fdata(cache_mode, dtype=np.float32)

        return input_data, gt_data

    def get_pair_slice(self, slice_index, slice_axis=2):
        """Return the specified slice from (input, ground truth).
        :param slice_index: the slice number.
        :param slice_axis: axis to make the slicing.
        """
        if self.cache:
            input_dataobj, gt_dataobj = self.get_pair_data()
        else:
            # use dataobj to avoid caching
            input_dataobj = self.input_handle.dataobj

            if self.gt_handle is None:
                gt_dataobj = None
            else:
                gt_dataobj = self.gt_handle.dataobj

        if slice_axis not in [0, 1, 2]:
            raise RuntimeError("Invalid axis, must be between 0 and 2.")

        if slice_axis == 2:
            input_slice = np.asarray(input_dataobj[..., slice_index],
                                     dtype=np.float32)
        elif slice_axis == 1:
            input_slice = np.asarray(input_dataobj[:, slice_index, ...],
                                     dtype=np.float32)
        elif slice_axis == 0:
            input_slice = np.asarray(input_dataobj[slice_index, ...],
                                     dtype=np.float32)

        # Handle the case for unlabeled data
        gt_meta_dict = None
        if self.gt_handle is None:
            gt_slice = None
        else:
            if slice_axis == 2:
                gt_slice = np.asarray(gt_dataobj[..., slice_index],
                                      dtype=np.float32)
            elif slice_axis == 1:
                gt_slice = np.asarray(gt_dataobj[:, slice_index, ...],
                                      dtype=np.float32)
            elif slice_axis == 0:
                gt_slice = np.asarray(gt_dataobj[slice_index, ...],
                                      dtype=np.float32)

            gt_meta_dict = mt_datasets.SampleMetadata({
                "zooms": self.gt_handle.header.get_zooms()[:2],
                "data_shape": self.gt_handle.header.get_data_shape()[:2],
            })

        input_meta_dict = mt_datasets.SampleMetadata({
            "zooms": self.input_handle.header.get_zooms()[:2],
            "data_shape": self.input_handle.header.get_data_shape()[:2],
        })

        dreturn = {
            "input": input_slice,
            "gt": gt_slice,
            "input_metadata": input_meta_dict,
            "gt_metadata": gt_meta_dict,
        }

        return dreturn
    
class MRI2DSegmentationDataset(Dataset):
    """This is a generic class for 2D (slice-wise) segmentation datasets.
    :param filename_pairs: a list of tuples in the format (input filename,
                           ground truth filename).
    :param slice_axis: axis to make the slicing (default axial).
    :param cache: if the data should be cached in memory or not.
    :param transform: transformations to apply.
    """
    def __init__(self, filename_pairs, slice_axis=2, p=1, cache=True,
                 transform=None, preprocess=None, slice_filter_fn=None, canonical=False):
        self.filename_pairs = filename_pairs
        self.handlers = []
        self.indexes = []
        self.transform = transform
        self.preprocess = preprocess
        self.cache = cache
        self.slice_axis = slice_axis
        self.slice_filter_fn = slice_filter_fn
        self.canonical = canonical
        self.p = p

        self._load_filenames()
        self._prepare_indexes()

    def _load_filenames(self):
        for input_filename, gt_filename in self.filename_pairs:
            segpair = SegmentationPair2D(input_filename, gt_filename,
                                         self.cache, self.canonical)
            self.handlers.append(segpair)

    def _prepare_indexes(self):
        for segpair in self.handlers:
            input_data_shape, _ = segpair.get_pair_shapes()
            for segpair_slice in range(input_data_shape[2]):

                # Check if slice pair should be used or not
                if self.slice_filter_fn:
                    slice_pair = segpair.get_pair_slice(segpair_slice,
                                                        self.slice_axis)

                    filter_fn_ret = self.slice_filter_fn(slice_pair)
                    if not filter_fn_ret:
                        continue

                item = (segpair, segpair_slice)
                self.indexes.append(item)

    def set_transform(self, transform):
        """This method will replace the current transformation for the
        dataset.
        :param transform: the new transformation
        """
        self.transform = transform

    def compute_mean_std(self, verbose=False):
        """Compute the mean and standard deviation of the entire dataset.
        :param verbose: if True, it will show a progress bar.
        :returns: tuple (mean, std dev)
        """
        sum_intensities = 0.0
        numel = 0

        with mt_datasets.DatasetManager(self,
                            override_transform=mt_transforms.ToTensor()) as dset:
            pbar = tqdm(dset, desc="Mean calculation", disable=not verbose)
            for sample in pbar:
                input_data = sample['input']
                sum_intensities += input_data.sum()
                numel += input_data.numel()
                pbar.set_postfix(mean="{:.2f}".format(sum_intensities / numel),
                                 refresh=False)

            training_mean = sum_intensities / numel

            sum_var = 0.0
            numel = 0

            pbar = tqdm(dset, desc="Std Dev calculation", disable=not verbose)
            for sample in pbar:
                input_data = sample['input']
                sum_var += (input_data - training_mean).pow(2).sum()
                numel += input_data.numel()
                pbar.set_postfix(std="{:.2f}".format(np.sqrt(sum_var / numel)),
                                 refresh=False)

        training_std = np.sqrt(sum_var / numel)
        return training_mean.item(), training_std.item()

    def __len__(self):
        """Return the dataset size."""
        return len(self.indexes)

    def __getitem__(self, index):
        """Return the specific index pair slices (input, ground truth).
        :param index: slice index.
        """
        segpair, segpair_slice = self.indexes[index]
        pair_slice = segpair.get_pair_slice(segpair_slice,
                                            self.slice_axis)

        # Consistency with torchvision, returning PIL Image
        # Using the "Float mode" of PIL, the only mode
        # supporting unbounded float32 values
        input_img = Image.fromarray(pair_slice["input"], mode='F')

        # Handle unlabeled data
        if pair_slice["gt"] is None:
            gt_img = None
        else:
            gt_img = Image.fromarray(pair_slice["gt"], mode='F')

        data_dict = {
            'input': input_img,
            'gt': gt_img,
            'input_metadata': pair_slice['input_metadata'],
            'gt_metadata': pair_slice['gt_metadata'],
        }

        # Changed: augment only a probability p of the data
        if self.transform is not None:
            if np.random.random() < self.p:
                data_dict = self.transform(data_dict)
            else:
                if self.preprocess is not None:
                    data_dict = self.preprocess(data_dict)
        else:
            if self.preprocess is not None:
                data_dict = self.preprocess(data_dict)

        return data_dict
    
### MULTI CHANNEL ###
class MultiChannelSegmentationPair2D(object):
    """This class is used to build multi-channel 2D segmentation datasets. 
    It represents a pair of of two data volumes (the input data and the ground 
                                                 truth data).
    :param input_filename: the input filename (supported by nibabel).
    :param gt_filename: the ground-truth filename.
    :param cache: if the data should be cached in memory or not.
    :param canonical: canonical reordering of the volume axes.
    """
    def __init__(self, input_filename, gt_filename, cache=True,
                 canonical=False):
        self.input_filename = input_filename
        self.gt_filename = gt_filename
        self.canonical = canonical
        self.cache = cache

        self.input_handle = nib.load(self.input_filename)

        # Unlabeled data (inference time)
        if self.gt_filename is None:
            self.gt_handle = None
        else:
            self.gt_handle = nib.load(self.gt_filename)

        if len(self.input_handle.shape) > 3:
            # Changed: instead of throwing a warning, change the dimension
            # of the input
            input_handle = self.input_handle
            
            # Changed: multiple input handles
            if input_handle.shape[3] == 80:
                self.input_handle = nib.funcs.four_to_three(input_handle)[0] # for bvalue=0
                self.input_handle1 = nib.funcs.four_to_three(input_handle)[4] # for bvalue=700
                # self.input_handle2 = nib.funcs.four_to_three(input_handle)[24] ## for bvalue=10
                
            elif input_handle.shape[3] == 60:
                self.input_handle = nib.funcs.four_to_three(input_handle)[0] # for bvalue=0
                self.input_handle1 = nib.funcs.four_to_three(input_handle)[10] # for bvalue=700
                # self.input_handle2 = nib.funcs.four_to_three(input_handle)[18] # for bvalue=10

        # Sanity check for dimensions, should be the same
        input_shape, gt_shape = self.get_pair_shapes()

        if self.gt_handle is not None:
            if not np.allclose(input_shape, gt_shape):
                raise RuntimeError('Input and ground truth with different dimensions.')

        if self.canonical:
            self.input_handle = nib.as_closest_canonical(self.input_handle)

            # Unlabeled data
            if self.gt_handle is not None:
                self.gt_handle = nib.as_closest_canonical(self.gt_handle)

    def get_pair_shapes(self):
        """Return the tuple (input, ground truth) representing both the input
        and ground truth shapes."""
        input_shape = self.input_handle.header.get_data_shape()

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_shape = None
        else:
            gt_shape = self.gt_handle.header.get_data_shape()

        return input_shape, gt_shape

    def get_pair_data(self):
        """Return the tuble (input, ground truth) with the data content in
        numpy array."""
        cache_mode = 'fill' if self.cache else 'unchanged'
        
        # Changed: stack multiple input channels on top of each other
        input_data1 = self.input_handle.get_fdata(cache_mode, dtype=np.float32)
        input_data2 = self.input_handle1.get_fdata(cache_mode, dtype=np.float32)
        # input_data3 = self.input_handle2.get_fdata(cache_mode, dtype=np.float32)

        input_data = np.concatenate([[input_data1, input_data2]], axis=3)

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_data = None
        else:
            gt_data = self.gt_handle.get_fdata(cache_mode, dtype=np.float32)

        return input_data, gt_data

    def get_pair_slice(self, slice_index, slice_axis=2):
        """Return the specified slice from (input, ground truth).
        :param slice_index: the slice number.
        :param slice_axis: axis to make the slicing.
        """
        
        if self.cache:
            input_dataobjs, gt_dataobj = self.get_pair_data()
        else:
            # use dataobj to avoid caching
            input_dataobj = self.input_handle.dataobj

            if self.gt_handle is None:
                gt_dataobj = None
            else:
                gt_dataobj = self.gt_handle.dataobj

        if slice_axis not in [0, 1, 2]:
            raise RuntimeError("Invalid axis, must be between 0 and 2.")

        input_slices = []
        for input_dataobj in input_dataobjs: 
            if slice_axis == 2:
                input_slice = np.asarray(input_dataobj[..., slice_index],
                                         dtype=np.float32)
                input_slices.append(input_slice)
            elif slice_axis == 1:
                input_slice = np.asarray(input_dataobj[:, slice_index, ...],
                                         dtype=np.float32)
            elif slice_axis == 0:
                input_slice = np.asarray(input_dataobj[slice_index, ...],
                                         dtype=np.float32)

        input_slice = np.stack(input_slices)

        # Handle the case for unlabeled data
        gt_meta_dict = None
        if self.gt_handle is None:
            gt_slice = None
        else:
            if slice_axis == 2:
                gt_slice = np.asarray(gt_dataobj[..., slice_index],
                                      dtype=np.float32)
            elif slice_axis == 1:
                gt_slice = np.asarray(gt_dataobj[:, slice_index, ...],
                                      dtype=np.float32)
            elif slice_axis == 0:
                gt_slice = np.asarray(gt_dataobj[slice_index, ...],
                                      dtype=np.float32)

            gt_meta_dict = mt_datasets.SampleMetadata({
                "zooms": self.gt_handle.header.get_zooms()[:2],
                "data_shape": self.gt_handle.header.get_data_shape()[:2],
            })

        input_meta_dict = mt_datasets.SampleMetadata({
            "zooms": self.input_handle.header.get_zooms()[:2],
            "data_shape": self.input_handle.header.get_data_shape()[:2],
        })

        dreturn = {
            "input": input_slice,
            "gt": gt_slice,
            "input_metadata": input_meta_dict,
            "gt_metadata": gt_meta_dict,
        }

        return dreturn
    
class MultiChannelMRI2DSegmentationDataset(Dataset):
    """This is a generic class for multi-channel 2D (slice-wise) segmentation 
    datasets.
    :param filename_pairs: a list of tuples in the format (input filename,
                           ground truth filename).
    :param slice_axis: axis to make the slicing (default axial).
    :param cache: if the data should be cached in memory or not.
    :param transform: transformations to apply.
    """
    def __init__(self, filename_pairs, slice_axis=2, p=1, cache=True,
                 transform=None, preprocess=None, slice_filter_fn=None, canonical=False):
        self.filename_pairs = filename_pairs
        self.handlers = []
        self.indexes = []
        self.transform = transform
        self.preprocess = preprocess
        self.cache = cache
        self.slice_axis = slice_axis
        self.slice_filter_fn = slice_filter_fn
        self.canonical = canonical
        self.p = p

        self._load_filenames()
        self._prepare_indexes()

    def _load_filenames(self):
        for input_filename, gt_filename in self.filename_pairs:
            segpair = MultiChannelSegmentationPair2D(input_filename, gt_filename,
                                         self.cache, self.canonical)
            self.handlers.append(segpair)

    def _prepare_indexes(self):
        for segpair in self.handlers:
            input_data_shape, _ = segpair.get_pair_shapes()
            for segpair_slice in range(input_data_shape[2]):

                # Check if slice pair should be used or not
                if self.slice_filter_fn:
                    slice_pair = segpair.get_pair_slice(segpair_slice,
                                                        self.slice_axis)

                    filter_fn_ret = self.slice_filter_fn(slice_pair)
                    if not filter_fn_ret:
                        continue

                item = (segpair, segpair_slice)
                self.indexes.append(item)

    def set_transform(self, transform):
        """This method will replace the current transformation for the
        dataset.
        :param transform: the new transformation
        """
        self.transform = transform

    def compute_mean_std(self, verbose=False):
        """Compute the mean and standard deviation of the entire dataset.
        :param verbose: if True, it will show a progress bar.
        :returns: tuple (mean, std dev)
        """
        sum_intensities = 0.0
        numel = 0

        with mt_datasets.DatasetManager(self,
                            override_transform=mt_transforms.ToTensor()) as dset:
            pbar = tqdm(dset, desc="Mean calculation", disable=not verbose)
            for sample in pbar:
                input_data = sample['input']
                sum_intensities += input_data.sum()
                numel += input_data.numel()
                pbar.set_postfix(mean="{:.2f}".format(sum_intensities / numel),
                                 refresh=False)

            training_mean = sum_intensities / numel

            sum_var = 0.0
            numel = 0

            pbar = tqdm(dset, desc="Std Dev calculation", disable=not verbose)
            for sample in pbar:
                input_data = sample['input']
                sum_var += (input_data - training_mean).pow(2).sum()
                numel += input_data.numel()
                pbar.set_postfix(std="{:.2f}".format(np.sqrt(sum_var / numel)),
                                 refresh=False)

        training_std = np.sqrt(sum_var / numel)
        return training_mean.item(), training_std.item()

    def __len__(self):
        """Return the dataset size."""
        return len(self.indexes)

    def __getitem__(self, index):
        """Return the specific index pair slices (input, ground truth).
        :param index: slice index.
        """
        segpair, segpair_slice = self.indexes[index]
        pair_slice = segpair.get_pair_slice(segpair_slice,
                                            self.slice_axis)

        # Consistency with torchvision, returning PIL Image
        # Using the "Float mode" of PIL, the only mode
        # supporting unbounded float32 values
        
        input_stack = []
        for channel_slice in pair_slice['input']:
            input_stack.append(Image.fromarray(channel_slice, mode='F'))
            
        input_img = np.stack(input_stack, axis=2)

        # Handle unlabeled data
        if pair_slice["gt"] is None:
            gt_img = None
        else:
            gt_img = Image.fromarray(pair_slice["gt"], mode='F')

        data_dict = {
            'input': input_img,
            'gt': gt_img,
            'input_metadata': pair_slice['input_metadata'],
            'gt_metadata': pair_slice['gt_metadata'],
        }

        # Changed: augment only a probability p of the data
        if self.transform is not None:
            if np.random.random() < self.p:
                data_dict = self.transform(data_dict)
            else:
                if self.preprocess is not None:
                    data_dict = self.preprocess(data_dict)
        else:
            if self.preprocess is not None:
                data_dict = self.preprocess(data_dict)

        return data_dict
    
    
### AVERAGE IMAGE ###
bval=np.array([0, 0, 0, 0, 700, 700, 700, 700, 1, 1, 1, 1, 5,  5,  5,  5,  
               100, 100, 100, 100, 300, 300, 300, 300, 10, 10, 10, 10, 0, 0, 
               0, 0, 20, 20, 20, 20, 500, 500, 500, 500, 50, 50, 50, 50, 40, 
               40, 40, 40, 30, 30, 30, 30, 150, 150, 150, 150, 75,  75,  75, 
               75,  0, 0, 0, 0, 600, 600, 600, 600, 200, 200, 200, 200, 400, 
               400, 400, 400, 2, 2, 2, 2])
bval2=np.array([0, 700, 700, 700, 1, 1, 1, 0, 5, 5, 5, 100, 100, 100, 0, 300, 
                300, 300, 10, 10, 10, 0, 20, 20, 20, 500, 500, 500, 0, 50, 50,
                50, 40, 40, 40, 30, 30, 30, 0, 150, 150, 150, 75, 75, 75, 0, 
                600, 600, 600, 200, 200, 200, 0, 400, 400, 400, 2, 2, 2, 0, ])

class AvgImgSegmentationPair2D(object):
    """This class is used to build an Average Image 2D segmentation datasets. 
    It represents a pair of of two data volumes (the input data and the ground 
                                                 truth data).
    :param input_filename: the input filename (supported by nibabel).
    :param gt_filename: the ground-truth filename.
    :param cache: if the data should be cached in memory or not.
    :param canonical: canonical reordering of the volume axes.
    """
    def __init__(self, input_filename, gt_filename, cache=True,
                 canonical=False):
        self.input_filename = input_filename
        self.gt_filename = gt_filename
        self.canonical = canonical
        self.cache = cache

        self.input_handle = nib.load(self.input_filename)
        self.ims = []

        # Unlabeled data (inference time)
        if self.gt_filename is None:
            self.gt_handle = None
        else:
            self.gt_handle = nib.load(self.gt_filename)

        if len(self.input_handle.shape) > 3:
            # Changed: instead of throwing a warning, change the dimension
            # of the input
            
            # Changed: make a list of same b-value images
            if self.input_handle.shape[3] == 80:
                for i in np.where(bval==0)[0][:4]:
                    im = nib.funcs.four_to_three(self.input_handle)[i]
                    self.ims.append(im)

            elif self.input_handle.shape[3] == 60:
                for i in np.where(bval2==0)[0][:4]:
                    im = nib.funcs.four_to_three(self.input_handle)[i]
                    self.ims.append(im)
                    
            self.input_handle = nib.funcs.four_to_three(self.input_handle)[0]
                

        # Sanity check for dimensions, should be the same
        input_shape, gt_shape = self.get_pair_shapes()

        if self.gt_handle is not None:
            if not np.allclose(input_shape, gt_shape):
                raise RuntimeError('Input and ground truth with different dimensions.')

        if self.canonical:
            self.input_handle = nib.as_closest_canonical(self.input_handle)

            # Unlabeled data
            if self.gt_handle is not None:
                self.gt_handle = nib.as_closest_canonical(self.gt_handle)

    def get_pair_shapes(self):
        """Return the tuple (input, ground truth) representing both the input
        and ground truth shapes."""
        input_shape = self.input_handle.header.get_data_shape()

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_shape = None
        else:
            gt_shape = self.gt_handle.header.get_data_shape()

        return input_shape, gt_shape

    def get_pair_data(self):
        """Return the tuble (input, ground truth) with the data content in
        numpy array."""
        cache_mode = 'fill' if self.cache else 'unchanged'
        
        # Changed: take the average image of the list
        ims = np.array([np.array(im.get_fdata(cache_mode, dtype=np.float32)) for im in self.ims])
        input_data = np.average(ims, axis=0)

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_data = None
        else:
            gt_data = self.gt_handle.get_fdata(cache_mode, dtype=np.float32)

        return input_data, gt_data

    def get_pair_slice(self, slice_index, slice_axis=2):
        """Return the specified slice from (input, ground truth).
        :param slice_index: the slice number.
        :param slice_axis: axis to make the slicing.
        """
        if self.cache:
            input_dataobj, gt_dataobj = self.get_pair_data()
        else:
            # use dataobj to avoid caching
            input_dataobj = self.input_handle.dataobj

            if self.gt_handle is None:
                gt_dataobj = None
            else:
                gt_dataobj = self.gt_handle.dataobj

        if slice_axis not in [0, 1, 2]:
            raise RuntimeError("Invalid axis, must be between 0 and 2.")

        if slice_axis == 2:
            input_slice = np.asarray(input_dataobj[..., slice_index],
                                     dtype=np.float32)
        elif slice_axis == 1:
            input_slice = np.asarray(input_dataobj[:, slice_index, ...],
                                     dtype=np.float32)
        elif slice_axis == 0:
            input_slice = np.asarray(input_dataobj[slice_index, ...],
                                     dtype=np.float32)

        # Handle the case for unlabeled data
        gt_meta_dict = None
        if self.gt_handle is None:
            gt_slice = None
        else:
            if slice_axis == 2:
                gt_slice = np.asarray(gt_dataobj[..., slice_index],
                                      dtype=np.float32)
            elif slice_axis == 1:
                gt_slice = np.asarray(gt_dataobj[:, slice_index, ...],
                                      dtype=np.float32)
            elif slice_axis == 0:
                gt_slice = np.asarray(gt_dataobj[slice_index, ...],
                                      dtype=np.float32)

            gt_meta_dict = mt_datasets.SampleMetadata({
                "zooms": self.gt_handle.header.get_zooms()[:2],
                "data_shape": self.gt_handle.header.get_data_shape()[:2],
            })

        input_meta_dict = mt_datasets.SampleMetadata({
            "zooms": self.input_handle.header.get_zooms()[:2],
            "data_shape": self.input_handle.header.get_data_shape()[:2],
        })

        dreturn = {
            "input": input_slice,
            "gt": gt_slice,
            "input_metadata": input_meta_dict,
            "gt_metadata": gt_meta_dict,
        }

        return dreturn
    
class AvgImgMRI2DSegmentationDataset(Dataset):
    """This is a generic class for Average Image 2D (slice-wise) segmentation datasets.
    :param filename_pairs: a list of tuples in the format (input filename,
                           ground truth filename).
    :param slice_axis: axis to make the slicing (default axial).
    :param cache: if the data should be cached in memory or not.
    :param transform: transformations to apply.
    """
    def __init__(self, filename_pairs, slice_axis=2, p=1, cache=True,
                 transform=None, preprocess=None, slice_filter_fn=None, canonical=False):
        self.filename_pairs = filename_pairs
        self.handlers = []
        self.indexes = []
        self.transform = transform
        self.preprocess = preprocess
        self.cache = cache
        self.slice_axis = slice_axis
        self.slice_filter_fn = slice_filter_fn
        self.canonical = canonical
        self.p = p

        self._load_filenames()
        self._prepare_indexes()

    def _load_filenames(self):
        for input_filename, gt_filename in self.filename_pairs:
            segpair = AvgImgSegmentationPair2D(input_filename, gt_filename,
                                         self.cache, self.canonical)
            self.handlers.append(segpair)

    def _prepare_indexes(self):
        for segpair in self.handlers:
            input_data_shape, _ = segpair.get_pair_shapes()
            for segpair_slice in range(input_data_shape[2]):

                # Check if slice pair should be used or not
                if self.slice_filter_fn:
                    slice_pair = segpair.get_pair_slice(segpair_slice,
                                                        self.slice_axis)

                    filter_fn_ret = self.slice_filter_fn(slice_pair)
                    if not filter_fn_ret:
                        continue

                item = (segpair, segpair_slice)
                self.indexes.append(item)

    def set_transform(self, transform):
        """This method will replace the current transformation for the
        dataset.
        :param transform: the new transformation
        """
        self.transform = transform

    def compute_mean_std(self, verbose=False):
        """Compute the mean and standard deviation of the entire dataset.
        :param verbose: if True, it will show a progress bar.
        :returns: tuple (mean, std dev)
        """
        sum_intensities = 0.0
        numel = 0

        with mt_datasets.DatasetManager(self,
                            override_transform=mt_transforms.ToTensor()) as dset:
            pbar = tqdm(dset, desc="Mean calculation", disable=not verbose)
            for sample in pbar:
                input_data = sample['input']
                sum_intensities += input_data.sum()
                numel += input_data.numel()
                pbar.set_postfix(mean="{:.2f}".format(sum_intensities / numel),
                                 refresh=False)

            training_mean = sum_intensities / numel

            sum_var = 0.0
            numel = 0

            pbar = tqdm(dset, desc="Std Dev calculation", disable=not verbose)
            for sample in pbar:
                input_data = sample['input']
                sum_var += (input_data - training_mean).pow(2).sum()
                numel += input_data.numel()
                pbar.set_postfix(std="{:.2f}".format(np.sqrt(sum_var / numel)),
                                 refresh=False)

        training_std = np.sqrt(sum_var / numel)
        return training_mean.item(), training_std.item()

    def __len__(self):
        """Return the dataset size."""
        return len(self.indexes)

    def __getitem__(self, index):
        """Return the specific index pair slices (input, ground truth).
        :param index: slice index.
        """
        segpair, segpair_slice = self.indexes[index]
        pair_slice = segpair.get_pair_slice(segpair_slice,
                                            self.slice_axis)

        # Consistency with torchvision, returning PIL Image
        # Using the "Float mode" of PIL, the only mode
        # supporting unbounded float32 values
        input_img = Image.fromarray(pair_slice["input"], mode='F')

        # Handle unlabeled data
        if pair_slice["gt"] is None:
            gt_img = None
        else:
            gt_img = Image.fromarray(pair_slice["gt"], mode='F')

        data_dict = {
            'input': input_img,
            'gt': gt_img,
            'input_metadata': pair_slice['input_metadata'],
            'gt_metadata': pair_slice['gt_metadata'],
        }

        # Changed: augment only a probability p of the data
        if self.transform is not None:
            if np.random.random() < self.p:
                data_dict = self.transform(data_dict)
            else:
                if self.preprocess is not None:
                    data_dict = self.preprocess(data_dict)           
        else:
            if self.preprocess is not None:
                data_dict = self.preprocess(data_dict)

        return data_dict
 
    
    