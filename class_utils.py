#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pickle
from sklearn.decomposition import PCA

print('Import class utils: done')
#%%

class AugmentedMLP(nn.Module):
    def __init__(self, 
                 insize = 60, 
                 hiddensize = 512, 
                 outsize= None, 
                 layer_nb = 8,
                 batch_norm = True,
                 bn_output = True):
        """
        Create a MLP network made of Linear, LeakyReLU and BatchNorm layers. 
        The input, hidden, output sizes can be specified as well as the number of layers.
        
        
        Parameters
        ----------
        insize : int, optional
            Input size of the network. The default is 60.
        middlesize : int, optional
            Hidden size of the network. The default is 512.
        outsize : int, optional
            Output size of the network. The default (None) sets it equal to 'insize'.
        layer_nb : int, optional
            Number of layer of the network. The default is 8.
        batchnorm : bool, optional
            Indicates if we put BatchNorm layer or not. The default is True
        bn_output : bool, optional
            Tells if we apply a batch normalisation layer as the last layer of the network.
            The default is True.

        Returns
        -------
        None.

        """
        super(AugmentedMLP, self).__init__()
        
        self.mlp = nn.ModuleList()
        self.insize = insize
        self.hiddensize = hiddensize

        if outsize is not None:
            self.outsize = outsize
        else:
            self.outsize = insize
        
        for k in range(layer_nb):
            if k==0:
                self.mlp.append(nn.Linear(self.insize, self.hiddensize, bias=True))
                self.mlp.append(nn.LeakyReLU())
                if batch_norm:
                    self.mlp.append(nn.BatchNorm1d(self.hiddensize))
                
            elif k==(layer_nb-1):
                self.mlp.append(nn.Linear(self.hiddensize, self.outsize, bias=True))
                if batch_norm and bn_output:
                    self.mlp.append(nn.BatchNorm1d(self.outsize))       
                
            else:
                self.mlp.append(nn.Linear(self.hiddensize, self.hiddensize, bias=True))
                self.mlp.append(nn.LeakyReLU())            
                if batch_norm:
                    self.mlp.append(nn.BatchNorm1d(self.hiddensize))             
        

    def forward(self, x):
        """
        Applies the network to the argument 'x'

        Parameters
        ----------
        x : torch.Tensor
            Input of the network.

        Returns
        -------
        x : torch.Tensor
            output of the network.

        """
        for k in range(len(self.mlp)):
            x = self.mlp[k](x)
                                         
        return x
    
   
class Disentangler(nn.Module):
    def __init__(self, G_path='./models/ffhq.pkl', 
                 layer_nb = 8, 
                 batch_norm=True,
                 latent_size = None,
                 knee=60,
                 pca_fit=True):
        """
        Auto-encoder made to disentangle the StyleGAN's latent space.
        The new latent space created is called 'C'.

        Parameters
        ----------
        G_path : str, optional
            Path where to find the '.pkl' file containing the StyleGAN's weights.
            The default is './models/ffhq.pkl'.
        layer_nb : int, optional
            Layer number for both the encoder and the decoder. The default is 8.
        batch_norm : bool, optional
            Tells if we put batch normalisation layers in the encoder and decoders. The default is True.
        latent_size : int, optional
            Size of the C latent space. The default is None.
        knee : int, optional
            Indicates of many dimensions of the W_pca space we take as input of our auto-encoder.
            The default is 60.
        pca_fit : bool, optional
            Indicates if we compute the PCA. If you want to train the network, you must set it to True.
            On the other hand, if you import pretrained weights, set it to False.
            The default is True.

        Returns
        -------
        None.

        """
        super(Disentangler,self).__init__()
         

        with open(G_path,'rb') as f:
            G = pickle.load(f)['G'].cuda()
            
        zs = torch.randn(10000,512).cuda()
        ws = (G.mapping(zs,None)[:,0,:]).detach()
        pca = PCA(n_components=512)
        pca.fit(ws.cpu().numpy())
        
        if knee is None:
            self.knee = 60
        else:
            self.knee = knee

        del G, zs, ws #Free some space
        
        self.pca_mean = torch.nn.Parameter((torch.tensor(pca.mean_).cuda().detach()),requires_grad=False)
        self.pca_mat = torch.nn.Parameter((torch.tensor(pca.components_).cuda().detach()),requires_grad=False)
        del pca #Free some space
                    
        if latent_size is not None:
            self.latent_size = latent_size
        else:
            self.latent_size = self.knee
             
        self.Mapin = AugmentedMLP(insize=self.knee, hiddensize = 512, outsize= self.latent_size , layer_nb=layer_nb, batch_norm = batch_norm, bn_output = True) #Mapping network going from W to disentangled C space
        self.Mapout = AugmentedMLP(insize=self.latent_size,  hiddensize = 512, outsize=self.knee, layer_nb=layer_nb, batch_norm = batch_norm, bn_output = False) #Mapping network going from disentangled C space to W
    
    def encode(self,w):
        """
        Encodes the w argument into the C latent space

        Parameters
        ----------
        w : torch.Tensor
            Latent vector to encode.

        Raises
        ------
        ValueError
            Raises if the w's shape isn't of the following shape: (batch_size, nb_of_dimensions).

        Returns
        -------
        c : torch.Tensor
            new disentangled latent vector within the C latent space.

        """
        if len(w.shape) != 2:
            raise ValueError("w must be 2-dimensionnal, currently its shape is "+str(w.shape)+" whereas it should be of the form (batch_size,512)")
        w_pca = torch.matmul((w - self.pca_mean), self.pca_mat.T)
        
        w_pca_reduced = w_pca[:,:self.knee]
        c = self.Mapin(w_pca_reduced)
        return c
        
    def decode(self, c, w):
        """
        Decodes the c latent vector

        Parameters
        ----------
        c : torch.Tensor
            C latent vector you want to decode.
        w : torch.Tensor
            W latent vector corresponding to the one which generated c (E(w) = c).

        Raises
        ------
        ValueError
            Raises if the w's shape isn't of the following shape: (batch_size, nb_of_dimensions).

        Returns
        -------
        w_out : torch.Tensor
            Decoded vector which lies W.

        """
        if len(w.shape) != 2:
            raise ValueError("w must be 2-dimensionnal, currently its shape is "+str(w.shape)+" whereas it should be of the form (batch_size,512)")
        w_pca = torch.matmul((w - self.pca_mean), self.pca_mat.T)
        
        w_out_reduced = torch.cat((self.Mapout(c), w_pca[:,self.knee:]),1)

        w_out = torch.matmul(w_out_reduced,self.pca_mat) + self.pca_mean #Reverse PCA
        
        return w_out
            
    def forward(self, w):
        """
        Auto-encodes the w latent vector

        Parameters
        ----------
        w : torch.Tensor
            Latent vector to auto-encode.

        Raises
        ------
        ValueError
            Raises if the w's shape isn't of the following shape: (batch_size, nb_of_dimensions).

        Returns
        -------
        w_out : torch.Tensor
            Output of the auto-encoder.
        c : torch.Tensor
            Latent vector of the auto-encoder.

        """
        if len(w.shape) != 2:
            raise ValueError("w must be 2-dimensionnal, currently its shape is "+str(w.shape)+" whereas it should be of the form (batch_size,512)")
               
        w_pca = torch.matmul((w - self.pca_mean), self.pca_mat.T)        
        w_pca_reduced = w_pca[:,:self.knee]        
        c = self.Mapin(w_pca_reduced)
        
        w_out_reduced = torch.cat((self.Mapout(c), w_pca[:,self.knee:]),1)
        w_out = torch.matmul(w_out_reduced,self.pca_mat) + self.pca_mean
        
        return w_out, c
    
    def cuda(self):
        self.pca_mean = self.pca_mean.cuda().detach()
        self.pca_mat = self.pca_mat.cuda().detach()
        self.Mapin = self.Mapin.cuda()
        self.Mapout = self.Mapout.cuda()
    
    def cpu(self):
        self.pca_mean = self.pca_mean.cpu()
        self.pca_mat = self.pca_mat.cpu()
        self.Mapin = self.Mapin.cpu()
        self.Mapout = self.Mapout.cpu()
             

class LADataset(Dataset):
    def __init__(self, data_file = './data/latent_attributes_dataset_gauss/data.pkl', label_file='./data/latent_attributes_dataset_gauss/label.pkl'):
        """
        Creates a class representing the (latent vectors, attributes) dataset.

        Parameters
        ----------
        data_file : str or list of str, optional
            Pickle file(s) where the latent vectors are stored. It should be of shape (Nb_samples, 512).
            The default is './data/latent_attributes_dataset_gauss/data.pkl'.
        label_file : str or list of str, optional
            Pickle file(s) where the attributes corresponding to the associated latent vectors are stored.
            It should be of shape (Nb_samples, Nb_attributes).
            The default is './data/latent_attributes_dataset_gauss/label.pkl'.

        Returns
        -------
        None.

        """
        super(LADataset,self).__init__()
        
        if label_file.__class__ == str:
            self.labels = torch.tensor(pd.read_pickle(label_file).to_numpy())
        elif label_file.__class__ == list:
            self.labels = torch.tensor([])
            for path in label_file:
                tmp = torch.load(path).float()
                self.labels = torch.cat((self.labels,tmp),axis=0)
            
        if data_file.__class__ == str:
            self.data = torch.tensor(pd.read_pickle(data_file).to_numpy())
        elif data_file.__class__ == list:
            self.data = torch.tensor([])
            for path in data_file:
                tmp = torch.load(path).float()
                self.data = torch.cat((self.data,tmp.reshape(tmp.shape[0],-1)),axis=0)
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        data = (self.data)[i,:]
        label = (self.labels)[i,:]
        return data, label
    
    def to(self, device):
        (self.labels) = (self.labels).to(device)
        (self.data) = (self.data).to(device)
        
        
class TrainingLoss(nn.Module):
    def __init__(self, attributes, attr_weight=1, dis_weight = 1, disentanglement_loss='Correlation'):
        """
        Loss function used to train our network. It is made of 3 terms:
            Reconstruction term 
            Attribute term
            Disentanglement term

        Parameters
        ----------
        attributes : torch.Tensor
            Tensor indicating which attributes have to be taken into account in the dataset. 
            Indeed, you may only want to consider a subset of the attributes you have in your LADataset.
        attr_weight : float, optional
            Weight for attribute term. The default is 1.
        dis_weight : float, optional
            Weight for disentanglement term. The default is 1.
        disentanglement_loss : str, optional
            Type of disentanglement loss to use. You can force disentanglement using 'Correlation'.
            Or you can take natural correlations between attr. into account using 'CorrelationPenalised'.
            The default is 'Correlation'.

        Returns
        -------
        None.

        """
        super(TrainingLoss, self).__init__()
        self.attr_weight = attr_weight
        self.dis_weight = dis_weight
        
        self.disentanglement_loss = DisentanglementLoss(attributes, loss_type=disentanglement_loss)
        self.attribute_loss = AttributeLoss(attributes)
        self.reconstruction_loss = ReconstructionLoss()
        
    def forward(self, inputs, outputs, codes, labels):
        """
        Computes the loss

        Parameters
        ----------
        inputs : torch.Tensor
            Dataset W latent vectors.
        outputs : torch.Tensor
            Autoencoder outputs, these vectors lie in W.
        codes : torch.Tensor
            C latent vectors
        labels : torch.Tensor
            Attribute values associated to latent vectors 'input' in the dataset.

        Returns
        -------
        float
            Loss value.

        """
        
        ##### Attribute term
        attr_loss = self.attribute_loss(codes, labels)

        ##### Disentanglement term
        dis_loss = self.disentanglement_loss(codes)

        ##### Recontruction term
        recons_loss = self.reconstruction_loss(inputs,outputs)
        
        return recons_loss + self.attr_weight * attr_loss + self.dis_weight * dis_loss


class ReconstructionLoss(nn.Module):
    def __init__(self):
        """
        Reconstruction Loss (L2 Norm)

        Returns
        -------
        None.

        """
        super(ReconstructionLoss, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, inputs, outputs):
         mse_val = self.mse(inputs,outputs)
         return mse_val

    
class AttributeLoss(nn.Module):
    def __init__(self, attributes):
        """
        Attribute Loss, it computes the L2 norm between latent vectors c and attributes

        Parameters
        ----------
        attributes : torch.Tensor
            Tensor indicating which attributes have to be taken into account in the dataset. 
            Indeed, you may only want to consider a subset of the attributes you have in your LADataset.

        Returns
        -------
        None.

        """
        super(AttributeLoss, self).__init__()
        self.attributes = attributes
        
    def forward(self, codes, labels):
        attr = torch.sum((codes[:,self.attributes]-labels[:,self.attributes])**2,axis=1)
        return torch.mean(attr)

          
class DisentanglementLoss(nn.Module):
    def __init__(self, attributes , loss_type='Correlation'):
        """
        Disentanglement Loss.

        Parameters
        ----------
        attributes : torch.Tensor
            Tensor indicating which attributes have to be taken into account in the dataset. 
            Indeed, you may only want to consider a subset of the attributes you have in your LADataset.
        loss_type : str, optional
            Type of disentanglement loss to use. You can force pure disentanglement using 'Correlation'.
            Or you can take natural correlations between attr. into account using 'CorrelationPenalised'.
            If you choose the latter, the natural correlations are represented as the CelebA attributes
            autocorrelation matrix.
            The default is 'Correlation'.

        Returns
        -------
        None.

        """
        super(DisentanglementLoss, self).__init__()
        self.attributes = attributes
        self.loss_type = loss_type
        
        attr = torch.tensor(pd.read_csv('./data/celeba/list_attr_celeba.csv').to_numpy()[:,1:].astype(np.float64))
        attr_normed = (attr- torch.mean(attr,axis=0))/torch.std(attr,axis=0)
        self.corr_celeba = (attr_normed.T @ attr_normed) / 202598.
        
    def forward(self, codes):
        c = codes
        c_stded = (c-torch.mean(c, dim=0))/torch.std(c,dim=0)
        corr_mat = (c_stded.T @ c_stded) / codes.shape[0]
        
        if self.loss_type == 'Correlation':
            return torch.sum(torch.abs(torch.eye(codes.shape[1]).cuda() - corr_mat))
        
        elif self.loss_type == 'CorrelationPenalised':
            reference = torch.eye(codes.shape[1])
            reference[self.attributes,:][:,self.attributes] = self.corr_celeba[self.attributes,:][:,self.attributes].float() 
            return torch.sum(torch.abs(reference.cuda() - corr_mat))
