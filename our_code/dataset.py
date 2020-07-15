import numpy as np
import torch
from provided_code.general_functions import get_paths, load_file
import torch.utils.data as data
from torch.nn.functional import interpolate, pad
from torch import nn
#from kornia import augmentation as aug
#import kornia
import pandas as pd
import copy
import matplotlib.pyplot as plt



#class transformation(nn.Module):
 #   def __init__(self) -> None:
  #      super(transformation, self).__init__()

   #     self.aff = aug.RandomAffine((-10.0, 10.0),
    #                                (0.1, 0.1), same_on_batch=True, resample=0)

     #   #self.crop = aug.RandomCrop((128, 128))

    #def forward(self, input, param=None):
     #       input = self.aff(input, param)

      #      return input

#from kornia.augmentation import AugmentationBase

#class MyRandomTransform(AugmentationBase):
 #   def __init__(self, return_transform: bool = False) -> None:
  #      super(MyRandomTransform, self).__init__(return_transform)
   #     self.angles_list = torch.Tensor([[40], [80], [120], [160], [200], [240], [280], [320]])

    #def generate_parameters(self, input_shape: torch.Size):
     #   # generate the random parameters for your use case.
      #  #angles_rad torch.Tensor = torch.rand(batch_shape) * kornia.pi
       # #angles_deg = kornia.rad2deg(angles_rad) * self.angle
        #if np.random.random() > 0.3:
     #       index = np.random.randint(0, 7)
     #       angles_deg = self.angles_list[index]

     #   else:
     #       angles_deg = torch.Tensor([0])
     #   return dict(angles=angles_deg)

   # def compute_transform(self, input, params):
    #    # compute transformation
   #     angles: torch.Tensor = params['angles'].type_as(input).repeat(input.shape[0])
   #     center = torch.tensor([[64, 64]]).type_as(input).repeat(input.shape[0], 1)
    #    transform = kornia.get_rotation_matrix2d(center, angles, torch.ones_like(angles))
    #    return transform

    #def apply_transform(self, input, params):
     #   # compute transformation
      #  transform = self.compute_transform(input, params)

       # # apply transformation and return
        #output = kornia.warp_affine(input, transform, (128, 128),'nearest')
        #return output

class CTDoseDataSet(data.Dataset):
    """Generates data for tensorflow"""

    def __init__(self, file_paths_list, batch_size=1, patient_shape=(128, 128, 128), shuffle=True,
                 mode_name='training_model', trans=[128]):
        """Initialize the DataLoader class, which loads the data for OpenKBP
        :param file_paths_list: list of the directories or single files where data for each patient is stored
        :param batch_size: the number of data points to lead in a single batch
        :param patient_shape: the shape of the patient data
        :param shuffle: whether or not order should be randomized
        """
        # Set file_loader specific attributes
        self.rois = dict(oars=['Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid',
                               'Esophagus', 'Larynx', 'Mandible'], targets=['PTV56', 'PTV63', 'PTV70'])

        #self.batch_size = batch_size  # Number of patients to load in a single batch
        self.patient_shape = patient_shape  # Shape of the patient
        self.indices = np.arange(len(file_paths_list))  # Indices of file paths
        self.file_paths_list = file_paths_list  # List of file paths
        self.shuffle = shuffle  # Indicator as to whether or not data is shuffled
        self.full_roi_list = sum(map(list, self.rois.values()), [])  # make a list of all rois
        self.num_rois = len(self.full_roi_list)
        self.patient_id_list = ['pt_{}'.format(k.split('/pt_')[1].split('/')[0].split('.csv')[0]) for k in
                                self.file_paths_list]  # the list of patient ids with information in this data loader
        self.trans = trans
        # Set files to be loaded
        self.required_files = None
       # self.aff_trans = MyRandomTransform()#transformation()
        self.mode_name = mode_name  # Defines the mode for which data must be loaded for
        self.set_mode(self.mode_name)  # Set load mode to prediction by default
        
        self.ct_scaling_factor = 4000.      #Added by Ramsy
        self.dose_scaling_factor = 100.     #Added by Ramsy

    def __getitem__(self, index):#get_batch(self, index=None, patient_list=None):
        """Loads one batch of data
        :param index: the index of the batch to be loaded
        :param patient_list: the list of patients for which to load data for
        :return: a dictionary with the loaded data
        """

        #if patient_list is None:
            # Generate batch based on the provided index
        indices = self.indices[index]# * self.batch_size:(index + 1) * self.batch_size]
        #else:
            # Generate batch based on the request patients
        #    indices = self.patient_to_index(patient_list)

        # Make a list of files to be loaded
        file_paths_to_load = [self.file_paths_list[indices]]#k] for k in

        # Load the requested files as a tensors
        loaded_data = self.load_data(file_paths_to_load)

        if self.trans != []:
            for transform in self.trans:
                if transform == 'flip':
                    if np.random.random() >= 0.5:
                        for key in loaded_data.keys():
                            if not isinstance(loaded_data[key], list):
                                if loaded_data[key].ndim == 4:

                                    loaded_data[key] = loaded_data[key][:, :, :, ::-1]
                                    if key == 'structure_masks':
                                        loaded_data[key][[3, 2]] = loaded_data[key][[2, 3]]

                elif isinstance(transform, int):
                    for key in loaded_data.keys():
                        if not isinstance(loaded_data[key], list):
                            if loaded_data[key].ndim == 4:
                                loaded_data[key] = interpolate(torch.from_numpy(loaded_data[key].copy()).unsqueeze(0), scale_factor=transform/128).squeeze(0).float()


                elif transform == 'crop':

                    seed_x = np.random.randint(0,10)
                    seed_y = np.random.randint(0,10)
                    seed_z = np.random.randint(0,10)
                    for key in loaded_data.keys():
                        if not isinstance(loaded_data[key], list):
                            if loaded_data[key].ndim == 4:
                                loaded_data[key] = pad(loaded_data[key], [10, 10, 10, 10, 10, 10], 'constant', 0)
                                loaded_data[key] = loaded_data[key][:, seed_x:128+seed_x, seed_y:128+seed_y, seed_z:128+seed_z]

            loaded_data['ct'] = ((loaded_data['ct'].clamp(0, self.ct_scaling_factor) /self.ct_scaling_factor))  # -0.5)/0.5
            if self.mode_name == 'training_model':
                loaded_data['dose'] = ((loaded_data['dose'].clamp(0, self.dose_scaling_factor) / self.dose_scaling_factor))  # -0.5)/0.5

        return loaded_data

    def patient_to_index(self, patient_list):
        """Converts a list of patient ids to their appropriate indices
        :param patient_list: list of patient ids
        :return: list of indices for the requested patients
        """
        # Get the indices for the list that is not shuffled
        un_shuffled_indices = [self.patient_id_list.index(k) for k in patient_list]

        # Map the indices to the shuffled indices to the shuffled indices
        shuffled_indices = [self.indices[k] for k in un_shuffled_indices]

        return shuffled_indices

    def set_mode(self, mode_name, single_file_name=None):
        """Selects the type of data that is loaded
        :param mode_name: the name of the mode that the data loader is switching to
        :param single_file_name: the name of the file that should be loaded (only used if the mode_name is 'single_file')
        """
        self.mode_name = mode_name

        if mode_name == 'training_model':
            # The mode that should be used when training or validing a model
            self.required_files = {'dose': ((1,) + self.patient_shape),  # The shape of dose tensor
                                   'ct': ((1,) + self.patient_shape),  # The shape of ct tensor
                                   'structure_masks': ((self.num_rois,) + self.patient_shape),
                                   # The shape of the structure mask tensor
                                   'possible_dose_mask': ((1,) + self.patient_shape),
                                   # Mask of where dose can be deposited
                                   'voxel_dimensions': (3,)
                                   # Physical dimensions (in mm) of voxels
                                   }
        elif mode_name == 'dose_prediction':
            # The mode that should be used when training or validing a model
            self.required_files = {'ct': ((1,) + self.patient_shape),  # The shape of ct tensor
                                   'structure_masks': ((self.num_rois,) + self.patient_shape),
                                   # The shape of the structure mask tensor
                                   'possible_dose_mask': ((1,) + self.patient_shape),
                                   # Mask of where dose can be deposited
                                   'voxel_dimensions': (3,)  # Physical dimensions (in mm) of voxels
                                   }
            self.batch_size = 1
            print('Warning: Batch size has been changed to 1 for dose prediction mode')

        elif mode_name == 'predicted_dose':
            # This mode loads a single feature (e.g., dose, masks for all structures)
            self.required_files = {mode_name: ((1,) + self.patient_shape)}  # The shape of a dose tensor

        elif mode_name == 'evaluation':
            # The mode that should be used evaluate the quality of predictions
            self.required_files = {'dose': ((1,) + self.patient_shape),  # The shape of dose tensor
                                   'structure_masks': ((self.num_rois,) + self.patient_shape),
                                   'voxel_dimensions': (3,),  # Physical dimensions (in mm) of voxels
                                   'possible_dose_mask': ((1,) + self.patient_shape),
                                   }
            self.batch_size = 1
            print('Warning: Batch size has been changed to 1 for evaluation mode')

        else:
            print('Mode does not exist. Please re-run with either \'training_model\', \'prediction\', '
                  '\'predicted_dose\', or \'evaluation\'')

    def number_of_batches(self):
        """Calculates how many full batches can be made in an epoch
        :return: the number of batches that can be loaded
        """
        return int(np.floor(len(self.file_paths_list) / self.batch_size))

    def on_epoch_end(self):
        """Randomizes the indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def load_data(self, file_paths_to_load):
        """Generates data containing batch_size samples X : (n_samples, *dim, n_channels)
        :param file_paths_to_load: the paths of the files to be loaded
        :return: a dictionary of all the loaded files
        """

        # Initialize dictionary for loaded data and lists to track patient path and ids
        tf_data = {}.fromkeys(self.required_files)
        patient_list = []
        patient_path_list = []

        # Loop through each key in tf data to initialize the tensor with zeros
        for key in tf_data:
            # Make dictionary with appropriate data sizes for bath learning
            tf_data[key] = np.zeros((self.required_files[key]))

        # Generate data
        for i, pat_path in enumerate(file_paths_to_load):
            # Get patient ID and location of processed data to load
            patient_path_list.append(pat_path)
            pat_id = pat_path.split('/')[-1].split('.')[0]
            patient_list.append(pat_id)
            # Make a dictionary of all the tensors
            loaded_data_dict = self.load_and_shape_data(pat_path)
            # Iterate through the dictionary add the loaded data to the "batch channel"
            for key in tf_data:
                if key != 'voxel_dimensions':
                    tf_data[key] = np.transpose(loaded_data_dict[key], [0,3,1,2])#[i,]
                else:
                    tf_data[key] = loaded_data_dict[key]


        # Add two keys to the tf_data dictionary to track patient information
        tf_data['patient_list'] = patient_list
        tf_data['patient_path_list'] = patient_path_list

        return tf_data

    def load_and_shape_data(self, path_to_load):
        """ Reshapes data that is stored as vectors into matrices
        :param path_to_load: the path of the data that needs to be loaded. If the path is a directory, all data in the
         directory will be loaded. If path is a file then only that file will be loaded.
        :return: Loaded data with the appropriate shape
        """

        # Initialize the dictionary for the loaded files
        loaded_file = {}
        if '.csv' in path_to_load:
            loaded_file[self.mode_name] = load_file(path_to_load)
        else:
            files_to_load = get_paths(path_to_load, ext='')
            # Load files and get names without file extension or directory
            for f in files_to_load:
                f_name = f.split('/')[-1].split('.')[0]
                if f_name in self.required_files or f_name in self.full_roi_list:
                    loaded_file[f_name] = load_file(f)

        # Initialize matrices for features
        shaped_data = {}.fromkeys(self.required_files)
        for key in shaped_data:
            shaped_data[key] = np.zeros(self.required_files[key])

        # Populate matrices that were no initialized as []
        for key in shaped_data:
            if key == 'structure_masks':
                # Convert dictionary of masks into a tensor (necessary for tensorflow)
                for roi_idx, roi in enumerate(self.full_roi_list):
                    if roi in loaded_file.keys():
                        np.put(shaped_data[key][roi_idx], loaded_file[roi], int(1))#self.num_rois * loaded_file[roi] + roi_idx
            elif key == 'possible_dose_mask':
                np.put(shaped_data[key], loaded_file[key], int(1))
            elif key == 'voxel_dimensions':
                shaped_data[key] = loaded_file[key]
            else:  # Files with shape
                np.put(shaped_data[key], loaded_file[key]['indices'], loaded_file[key]['data'])

        return shaped_data

    def get_transforms(self, data, trans):
        return data

    def __len__(self):
        return len(self.file_paths_list)
