import numpy as np
import torch
from statistics import mean


class DVH():
    
    
    def __init__(self, dataloader):
        
        self.oars = dataloader.dataset.rois['oars']
        self.targets = dataloader.dataset.rois['targets']
        self.full_roi_list = dataloader.dataset.full_roi_list
        self.dose_scaling_factor = dataloader.dataset.dose_scaling_factor



    def calculate_DVH_metrics(self, batch, prediction, detailed=False, shallow_dict=False ):
        #Calculates dose metrics from batch containing ct-images, structure data and ground truth
        #and the prediction as provided by the model. It is assumed that batch-size is set to one. 
        #Returns a dictionary containing the dose metrics as tensors
        #detailed:      If set to False, only the averages of the metrics D_99, D_95, D_1 and D_0.1cc, Mean 
        #               over the ROIs to which they apply and the DVH-Score as used in the openkbp-competition are calculated,
        #               with the names of the metrics as keys. If set to True, additionally to the above average values, the values of all dose metrics
        #               (except the DVH-Score) for all ROIs are also returned. 
        #shallow_dict:  Applys only if detailed is set to True. If shallow_dict is set to False every entry in the output dictionary is as dictionary
        #               containing the value for each ROI present in the patient, to which the metric applys (ROI-name is key) and the average value 
        #               of the metric ('average' is key)
        #               If set to True, the dictionary is flattened to one level of depth, for all metric values to be accessable via a 
        #               combined key.
        #               For the use in the context of the validation- and test-loop detailed has to be set to False, since not every patient,
        #               contains every ROI, which makes continuous evaluation of ROI-metrics for all ROIs impossbile. The other settings can
        #               be used to evaluate the metrics for single patients.
        
        DVH_metrics = {'D_99': {}, 'D_95': {}, 'D_1': {}, 'D_0.1_cc': {}, 'Mean': {}}  
         
        
        ct, dose, structure, voxel_dimensions = batch['ct'], batch['dose'], batch['structure_masks'], batch['voxel_dimensions']
        
        DVH_statistics_ground_truth = self.calculate_DVH_statistics(ct,dose,structure,voxel_dimensions)
        DVH_statistics_prediction = self.calculate_DVH_statistics(ct, prediction, structure, voxel_dimensions)
        
        for metric in DVH_statistics_ground_truth:
            for roi in DVH_statistics_ground_truth[metric]:
                DVH_metrics[metric][roi] = np.abs(DVH_statistics_prediction[metric][roi] - DVH_statistics_ground_truth[metric][roi])
                
            
        DVH_metrics_averages = {'D_99': {}, 'D_95': {}, 'D_1': {}, 'D_0.1_cc': {}, 'Mean': {}, 'DVH_Score':{}} 
        complete_metrics_list = []
        for metric in DVH_metrics:
            metrics_list = list(DVH_metrics[metric].values())
            complete_metrics_list.extend(metrics_list) 
            DVH_metrics_averages[metric] = torch.tensor(mean(metrics_list))
            
        DVH_metrics_averages['DVH_Score'] = torch.tensor(mean(complete_metrics_list))
            
                
        if detailed:
            DVH_metrics_output = {}
            for metric in DVH_metrics:
                for roi in DVH_metrics[metric]:
                    DVH_metrics[metric][roi] = torch.tensor(DVH_metrics[metric][roi])
                    if shallow_dict:
                        DVH_metrics_output[roi + '_' + metric] = DVH_metrics[metric][roi]
            if not shallow_dict:
                for metric in DVH_metrics:
                    DVH_metrics_output[metrics + '_' + 'average'] = DVH_metrics
            DVH_metrics_output.update(DVH_metrics_averages)
            
                
        else:
            DVH_metrics_output = DVH_metrics_averages
            
        return DVH_metrics_output
        
        




    def calculate_DVH_statistics(self,ct, dose, structure, voxel_dimensions):
    #Calulates the DVH-statistics for the given ct-image, structure data and dose.
    #Can be used to calculate the dose statistics for ground truth as well as prediction.
    #Returns a dictionary with the metrics D_99, D_95, D_1, D_0.1_cc and Mean as entries.
    #Every entry is a dictionary containing an entry for every ROI present in the given patient
    #to which the metric applies. 
    #The inputs to the function have to be tensors as contained in the batches supplied by
    #the dataloader of our model (with batch dimension )
    #The the values in the dictionary representing the dose statistics are numpy-arrays.
    
        ct, dose, structure, voxel_dimensions = ct.cpu().numpy(), dose.cpu().numpy()*self.dose_scaling_factor, structure.cpu().numpy(),voxel_dimensions.cpu().numpy()


        DVH_statistics = {'D_99': {}, 'D_95': {}, 'D_1': {}, 'D_0.1_cc': {}, 'Mean': {}}        #Dictionary containing containing dictionaries for the DVH-metrics of the
                                                                                                #current batch.


                                                   
        voxel_size = np.prod(voxel_dimensions)      #Volume of a voxel (in mm^3)
        voxels_in_tenth_of_cc = np.maximum(1, np.round(100/voxel_size))
        roi_exists = structure.max(axis=(0, 2, 3, 4))
        
        for roi_idx, roi in enumerate(self.full_roi_list):
            if roi_exists[roi_idx]:
                roi_mask = structure[:, roi_idx:roi_idx+1, :, :, :]
                roi_dose = dose[roi_mask.astype(bool)]
                roi_size = len(roi_dose)
                if roi in self.oars:
                    #Calculation of D_0.1_cc
                    fractional_volume_to_evaluate = 100 - voxels_in_tenth_of_cc / roi_size * 100
                    DVH_statistics['D_0.1_cc'][roi] = np.percentile(roi_dose, fractional_volume_to_evaluate)
                    #Calculation of mean
                    DVH_statistics['Mean'][roi] = roi_dose.mean().astype('float64')
                else:
                    #Calculation of D_99
                    DVH_statistics['D_99'][roi] = np.percentile(roi_dose, 1)
                    #Calculation of D_95
                    DVH_statistics['D_95'][roi] = np.percentile(roi_dose, 5)
                    #Calculation of D_1
                    DVH_statistics['D_1'][roi] = np.percentile(roi_dose, 99)
                    
        
        return DVH_statistics
                    
        
    
    
    
    
    
