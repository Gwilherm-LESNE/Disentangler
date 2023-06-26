#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import class_utils as cu
import pandas as pd
import os
import argparse

#%%
  
class DistributionTransformer():
    def __init__(self, dataset, N=None):
        """      

        Parameters
        ----------
        dataset : cu.LADataset
            Dataset you want to compute the histogram on.
            This histogram will be used to apply the histogram transfer later on.
            !!! IMPORTANT !!! We assume that the attribute values in 'dataset' are the ones taken 
            BEFORE sigmoid activation. Hence, they may (theorically) range from -inf. to +inf. 
        N : int, optional
            Integer indicating the number of samples taken from 'dataset' to compute the histogram.
            The default (None) is the size of 'dataset'.

        Returns
        -------
        None.

        """
        super(DistributionTransformer,self).__init__()
        
        if N is None:
            self.N = len(dataset)
        else:
            self.N = N
        
        rand_idx = np.random.randint(0,len(dataset),self.N)
        
        labels = dataset[rand_idx][1]
        labels[labels>40.] = 40. #Cropping attribute values
        labels[labels<-40.] = -40. #Cropping attribute values
        
        self.cumul = torch.zeros(40,201)
        self.bins = torch.zeros(40,201)
        
        for i in range(labels.shape[1]):
            hist,bns = torch.histogram(labels[:,i],bins=torch.linspace(-40.2,40.2,202))
            self.cumul[i,:] = torch.cumsum(hist,0)
            self.bins[i,:] = (bns[1:]+bns[:-1])/2
            
        self.cumul /= self.N
        
    def raw2Gaussian(self, data):
        """
        Transforms the raw values of the dataset into gaussianized ones.

        Parameters
        ----------
        data : cu.LADataset or torch.Tensor
            data on which you want to apply the histogram transfer.
            If torch.Tensor, we assume its shape being (Nb_samples, Nb_attributes).

        Raises
        ------
        ValueError
            If the 'data' argument isn't of the right type.

        Returns
        -------
        new_data : torch.Tensor
            Gaussianzed data having the same shape of 'data' argument.

        """
        if isinstance(data, cu.LADataset):
            data = data.labels
            
        if isinstance(data, torch.Tensor):
            data[data>40.] = 40.
            data[data<-40.] = -40.
            new_data = torch.zeros(data.shape)
            mat=torch.zeros(data.shape)
            
            if data.shape[0]>100000:
                print('Changing data distribution to a gaussian one')
                
            for i in range(data.shape[0]):
                if data.shape[0]>100000 and i%1000==0:
                    print('sample number '+str(i)+'/'+str(data.shape[0]))
                    
                for j in range(data.shape[1]):
                    tmp = torch.where(data[i,j]>self.bins[j])[0]
                    
                    if len(tmp)==0:
                        pos=0
                    elif len(tmp)==((self.bins).shape[1]):
                        pos = ((self.bins).shape[1])-2
                    else:
                        pos=torch.max(tmp)
                        
                    lamb = (data[i,j]-self.bins[j,pos])/(self.bins[j,pos+1]-self.bins[j,pos])
                    uni_val = self.cumul[j,pos] + lamb * (self.cumul[j,pos+1]-self.cumul[j,pos])
                    mat[i,j] = uni_val
                    
                    if uni_val == 1.:
                        new_data[i,j] = 3.1
                    elif uni_val == 0.:
                        new_data[i,j] = -3.1
                    else:
                        new_data[i,j] = (np.sqrt(2) * torch.erfinv(2*uni_val - 1))
                        
            return new_data
        else:
            raise ValueError(" 'data' doesn't have the right type, it is supposed to be Tensor or cu.LADataset instance while it's currently: "+ str(data.__class__))        
        
def create_gaussian_dataset(dataset, save_path = './data/latent_attributes_dataset_gauss/'):
    """
    Gaussianize a given dataset and store at the path provided in arguments.

    Parameters
    ----------
    dataset : cu.LADataset
        Dataset on which you want to apply the gaussianization process of its attribute values.
    save_path : str, optional
        PAth where to store the new gaussianzed dataset. The default is './data/latent_attributes_dataset_gauss/'.

    Returns
    -------
    None.

    """
    gaussianDataset = dataset
    transformer = DistributionTransformer(gaussianDataset)

    label = pd.DataFrame(transformer.raw2Gaussian(gaussianDataset))
    data = pd.DataFrame(gaussianDataset.data)
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    label.to_pickle(os.path.join(save_path,'label.pkl'))
    data.to_pickle(os.path.join(save_path,'data.pkl'))
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('dataset_folder', metavar = 'dataset_folder', help = "Folder path where to find the raw dataset files (data.pkl and label.pkl)", type=str)
    parser.add_argument("-s", '--save_path', metavar = 'save_path', help = "Path where to save the gaussianzed dataset", type=str,  default = './data/latent_attributes_dataset_gauss/')
    args = parser.parse_args()

    raw_dataset = cu.LADataset(data_file = os.path.join(args.dataset_folder,'data.pkl'), 
                                label_file = os.path.join(args.dataset_folder,'label.pkl'))
    
    create_gaussian_dataset(raw_dataset, save_path = args.save_path)
    