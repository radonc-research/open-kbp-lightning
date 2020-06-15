# OpenKBP Grand Challenge - best model of PTV group

![](read-me-images/aapm.png)
  
This is the network implementation of the Prediction Team Vienna for the _open-kbp_ grand challenge. The repository can be used only a local machine for now. 


 ![](read-me-images/pipeline.png)

## What this code does
  
  - _our_code/3D_loss.py_: includes a pretrained 3D model trained on kinetics video data used as for the feature loss.
  - _our_code/model_pix2pix.py_: Includes the basic U-Net model of pix2pix. Further options are available such as different 
   activation functions, normalisations, and ResNet blocks.
  - _train.py_: Is the main file from which the training takes place. You can directly alter the code from a editior e.g. PyCharm or
   you use the parser arguments over the command prompt.

## Requirements
Pytorch >=1.4

Pytorch Lightning >= 0.7.6

numpy >= 1.18.2

wandb >= 0.8.36

## Created folder structure
This repository will create a file structure that branches from a directory called _open-kbp_. The file structure
will keep information about predictions from a model (called baseline in this example) and the model itself in the
 _results_ directory. It assume that the data provided for the OpenKBP competition is in a directory called 
 _provided-data_. This code will also make a directory called _submissions_ to house the zip files that can be
  submitted to CodaLab for validation set evaluation (this code will generalize to test data once the test data is
   released). Use this folder tree as a reference (it will more or less build itself).
  