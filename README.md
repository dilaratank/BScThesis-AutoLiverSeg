# AutoLiverSeg
This repository includes the code of my Bachelor's thesis: Automatic Liver Segmentation of qMRI images:  A Deep Learning Approach. Below, you can find a summary of the project and a section describing the usage of the code. 

## Summary 
(Quantitative) magnetic resonance imaging ((q)MRI) is a great tool for imaging fatty livers. In qMRI, the intensity of each pixel in the image corresponds to a measurement of a real physical property. In researching the  assessment of disease severity along the NAFLD disease spectrum, this physical property is diffusion, as diffusion significantly correlates with disease activity. However, to obtain quantitative features, such as the diffusion, from medical images, regions of interest (ROIs) indicating the tissue of interest (e.g. liver in fatty liver disease) need to be drawn. Currently, this is mainly done by manual analysis, which is time-consuming, labor intensive and prone to error. A possible solution to this problem is using deep learning. For biomedical image segmentation, the U-Net architecture and its variations are widely used. While automatic liver segmentation has been proven to work well on CT and conventional MRI data, only a few studies can be found on automatic segmentation on qMRI data. We hypothesized that segmentation on qMRI data can be automated with a U-Net architecture. To investigate this, we trained a U-Net and some variations on data of 37 patients with NAFLD and 15 healthy volunteers (52 in total). The best performing model was the standard U-Net architecture, which received an mean Dice score of 0.91, with which we showed that automatic segmentation on qMRI data can indeed be automated. We propose that training a multi-channel U-Net with different b-values with augmentations as input could possibly even better this score. 

## Usage
### Preliminaries 
To be able to use this code, the following libraries must be installed: 
- MedicalTorch
- NumPy 
- Torch
- Scikit-learn

### Contents
The contents of this repository are as follows. 
- **model**: Contains the best performing model, mentioned in the Summary section above. 
- **notebooks**: Contains a visualization notebook, to visualize the dataset and the predictions of the model. 
- **predict**: Contains a tool that automatically segments the liver in a qMRI dataset. The tool was developed with the best performing model and some post-processing functions.
- **scripts**: Contains scripts that can be run with SLURM.
- **src**: Contains the code for training.
  - **utils**: Contains helper functions.

### Training
The training process is available for four different model architectures:
1. The conventional U-Net architecture ('unet')
2. The conventional SegNet architecture ('segnet')
3. A multi-channel U-Net architecture ('multichannel-unet')
4. An average image U-Net architecture ('avgimg-unet')
Also, there is an option for transfer learning for different masks. 

The four model architectures can be run like this: 
With SLURM:
```
sbatch baseline-unet.sh
```
Without SLURM:
```
python3 train.py --arch 'unet' --name 'unet'
```
Where ``` --arch 'unet' ``` can be replaced with the three other architectures. 

The transfer learning approach can be run like this:
With SLURM:
```
sbatch transfer.sh
```
Without SLURM:
```
python3 transfer_learning.py
```

### Prediction 
For instructions to use the prediction tool, please see the folder 'predictions'.
