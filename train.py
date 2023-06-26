#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")    
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import class_utils as cu
import sys
from itertools import product
from os.path import exists
import time
import argparse

print('Imports: done')
#%%

def tensor2Image(tensor): 
    """
    Converts a tensor of shape (n_channel, width, height) or (bsize, n_channel, width, height)
    to an image in numpy format (width, height,n_channels) or (bsize, width, height, n_channel)
    Applies clamping.

    Parameters
    ----------
    tensor : torch.Tensor
        input tensor to convert.

    Raises
    ------
    ValueError
        Raises if tensor isn't 4 or 3 dimensionnal.

    Returns
    -------
    np.Array
        Image corresponding to the input tensor.

    """
            
    shape = tensor.shape
    imgs=[]
    if len(shape)==4:
        for i in range(shape[0]):
            normalized_tensor = torch.clamp((0.5*tensor[i] + 0.5), 0, 1)
            temp_img = normalized_tensor.permute(1,2,0).numpy()
            imgs.append(temp_img)
        return np.array(imgs)
    elif len(shape)==3:
        normalized_tensor = torch.clamp((0.5*tensor + 0.5), 0, 1)
        tmp_img = normalized_tensor.permute(1,2,0).numpy()
        return tmp_img
    else:
        raise ValueError('tensor has wrong shape. It must be of type: (b_size,nb_channels,width,height) or (nb_channels,width,height)')

def w2Wp(w): 
    """
    Converts W latent vector to W+ latent vector by copying the same vector 18 times.
    Can handle batch_sizes

    Parameters
    ----------
    w : torch.Tensor or np.Array
        W latent vector.

    Returns
    -------
    torch.Tensor
        Corresponding W+ latent vector.

    """
    if isinstance(w, np.ndarray):
        w = torch.tensor(w)
    if len(w.shape)==3 and w.shape[1]==18 and w.shape[2]==512:
        return w
    if len(w.shape)==1:
        w = w.unsqueeze(0)
    wplus = w.unsqueeze(1).repeat(1,18,1)
    return wplus

def plot_results(targets, outputs):
    """
    Plotting function. The idea is to compare network outputs with labels side by side.

    Parameters
    ----------
    targets : torch.Tensor
        Reference torch images 
    outputs : torch.Tensor
        Output torch images to be compared to targets.

    Returns
    -------
    fig : plt.figure
        2*2 Figure where each cell represents the source image (left) and the network output image (right).

    """
    with open('./models/stylegan2.pkl','rb') as f:
        Gan = pickle.load(f)['G_ema'].cuda()
    
    if targets.shape[1]==512:
        w1 = w2Wp(targets[:4,:])
        w2 = w2Wp(outputs[:4,:])
    else:
        w1 = targets[:4]
        w2 = outputs[:4]
        w1 = w1.reshape(4,18,512)
        w2 = w2.reshape(4,18,512)
        
    plt.ioff()
    fig = plt.figure(figsize=(30, 60))
    
    for idx in np.arange(4):
        im1 = tensor2Image(Gan.synthesis(torch.unsqueeze(w1[idx],0).cuda(), noise_mode='const', force_fp32=True).detach().cpu())
        ax = fig.add_subplot(2, 4, 2*idx+1, xticks=[], yticks=[])
        ax.imshow(im1[0][0::4,0::4,::])
        ax.set_title("W n°"+str(idx)+", label")
        
        im2 = tensor2Image(Gan.synthesis(torch.unsqueeze(w2[idx],0).cuda(), noise_mode='const', force_fp32=True).detach().cpu())
        ax = fig.add_subplot(2, 4, 2*idx+2, xticks=[], yticks=[])
        ax.imshow(im2[0][0::4,0::4,::])
        ax.set_title("W n°"+str(idx)+", output")
        
    return fig

#-----------------------------------------------------------------------------#

def train(parameters, data_file, label_file, save_model=True, save_path='./models/disentangler/'):
    """
    Main function to train our network.

    Parameters
    ----------
    parameters : dict
        dictionnary containing all the hyperparameters.
    save_model : bool, optional
        Indicates if you want to save the trained model. The default is True.
    save_path : str, optional
        Path where to store the trained model, it will also be used to store tensorboard runs.
        The default is './models/disentangler/'.

    Returns
    -------
    None.

    """
    param_values = [v for v in parameters.values()]
    print('Parameters are :' +str(param_values))
    
    if not exists(save_path):
        os.mkdir(save_path)
    
    for run_id, params in enumerate(product(*param_values)):
        print('_'*50)
        print(' '*16+'RUN number ',run_id+1)
        print('_'*50)
        print(params)
        
        lr = params[0]
        b_size = params[1]
        n_epoch = params[2]
        layer_nb = params[3]
        attr_weight = params[4]
        dis_weight = params[5]
        batch_norm = params[6]
        attributes_idx = list(params[7])
        attributes_nb = params[8]
        disentanglement_loss = params[9]
        knee = params[10]
            
        torch.cuda.empty_cache() #empty cache of the GPU

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        sys.stdout.write('DEVICE TYPE :'+ device.type + '\n')
        sys.stdout.flush()        

        # --- Prepare data --- #
        dataset = cu.LADataset(data_file=data_file,label_file=label_file)
        
        train_size = int(0.7* len(dataset))
        test_size = len(dataset) - train_size
        
        training_data, test_data = torch.utils.data.random_split(dataset,(train_size,test_size))
        
        train_dataloader = DataLoader(training_data, batch_size = b_size, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_data, batch_size = b_size, shuffle=True, drop_last=True)
        
        
        # --- Define network and losses --- #
        net = cu.Disentangler(layer_nb = layer_nb,
                              batch_norm = batch_norm,
                              knee = knee)
        
        loss = cu.TrainingLoss(attributes = attributes_idx,
                               attr_weight = attr_weight,
                               dis_weight = dis_weight,                            
                               disentanglement_loss = disentanglement_loss)

        rloss = cu.ReconstructionLoss()
        aloss = cu.AttributeLoss(attributes_idx)
        dloss = cu.DisentanglementLoss(attributes_idx, loss_type=disentanglement_loss)
        
        optimizer = optim.Adam(net.parameters(), lr=lr)
        
        net.to(device)
    
        # --- Tensorboard set up --- #
        idx_name=0
        while exists(save_path+str(idx_name)):
            idx_name+=1
        os.mkdir(save_path+str(idx_name))
        writer = SummaryWriter(save_path+str(idx_name))
        
        # --- Train part --- #       
        sys.stdout.write('Starting Training'+'\n')
        sys.stdout.flush()
        
        for epoch in range(n_epoch):  # loop over the dataset multiple times
            
            print('-'*20)
            print(' '*5+'Epoch n°'+str(epoch+1))
            print('-'*20)
            sys.stdout.flush()
            
            running_loss = 0.0
            train_loss = 0.0
            train_steps = 0
            
            net.train(True)
                        
            for i, data in enumerate(train_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device).float(), data[1].to(device).float()
        
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize  
                outputs, codes = net(inputs)
                
                loss_value = loss(inputs, outputs, codes, labels)
                
                loss_value.backward()
                optimizer.step()
                
                # print statistics
                running_loss += loss_value.item()
                train_loss += loss_value.item()
                train_steps += 1
                
                if i % 100 == 99: # print every 100 mini-batches
                    print(f'[epoch n°{epoch + 1}, batch n°{i + 1:5d}] loss: {running_loss / 500:.3f}')
                    sys.stdout.flush()
                    running_loss = 0.0
            
            writer.add_scalar("Loss/train", train_loss/train_steps, epoch)
            writer.flush()
            
            # --- Test part --- #
            with torch.no_grad():
                
                test_loss = 0.0
                epoch_rloss = 0.0
                epoch_aloss = 0.0
                epoch_dloss = 0.0
                test_steps = 0

                net.eval()
                
                for i, data in enumerate(test_dataloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data[0].to(device).float(), data[1].to(device).float()
                    
                    # forward + loss computation                   
                    outputs, codes = net(inputs)
                                            
                    epoch_rloss += rloss(inputs,outputs).item()
                    epoch_aloss += aloss(codes,labels).item()
                    epoch_dloss += dloss(codes).item()
                    test_loss += loss(inputs,outputs,codes,labels).item()
                    test_steps+=1
                
                print(f"Test Loss 1 for epoch n°{epoch + 1}: {float(epoch_rloss/test_steps):.2f}")
                print(f"Test Loss 2 for epoch n°{epoch + 1}: {float(epoch_aloss/test_steps):.2f}")
                print(f"Test Loss 3 for epoch n°{epoch + 1}: {float(epoch_dloss/test_steps):.2f}")
                sys.stdout.flush()
                
                writer.add_scalar("Loss/ReconstructionLoss", epoch_rloss/test_steps, epoch)
                writer.add_scalar("Loss/AttributeLoss", epoch_aloss/test_steps, epoch)
                writer.add_scalar("Loss/DisentanglementLoss", epoch_dloss/test_steps, epoch)
                writer.add_scalar("Loss/TestLoss", test_loss/test_steps, epoch)
                writer.flush()
                
        # --- Figures plotting --- #  
        inputs = test_data[0:4][0].to(device).float()
        outputs, codes = net(inputs)
        outputs = outputs.detach()
        codes = codes.detach()

        writer.add_figure('data vs reconstructions',
                        plot_results(inputs, outputs),
                        global_step = epoch,
                        close=True)
        plt.close()
        
        idxs = torch.tensor(attributes_idx)[torch.randint(attributes_nb,(2,))]
        for i in range(4):
                new_val = -torch.sign(codes[i,idxs[i//2]])*2.5
                codes[i,idxs[i//2]] = new_val 
        
        edits = net.decode(codes.cuda(), inputs)
        writer.add_figure('Edit 2 attributes ('+str(idxs[0])+','+str(idxs[1])+')',
                        plot_results(inputs, edits),
                        global_step = epoch,
                        close=True)
        plt.close()
        
        # --- Hyperparameters saving --- #
        sys.stdout.write('Finished Training'+'\n')
        sys.stdout.flush()
        writer.add_hparams(
            {"lr": lr, 
             "bsize": b_size, 
             "layer_nb": layer_nb,
             "n_epoch": n_epoch,
             "batch_norm": batch_norm,
             "attr_weight" : attr_weight,
             "dis_weight": dis_weight,
             "attributes_idx": str(attributes_idx),
             "attributes_nb": attributes_nb,
             "disentanglement_loss": disentanglement_loss,
             "knee": knee},
            {"reconstruction loss": epoch_rloss/test_steps,
             "attribute loss": epoch_aloss/test_steps,
             "disentanglement loss": epoch_dloss/test_steps,
             "test loss": test_loss/test_steps,             
            },
        )
        
        # --- Network saving --- #
        save_name = save_path+str(idx_name)+'/disentangler'+str(idx_name)+'.pt'
    
        if save_model:
            if not exists(save_path+str(idx_name)):
                os.mkdir(save_path+str(idx_name))
            torch.save(net.state_dict(), save_name)
        
        
    writer.close()

    
def main_function(args):
    parameters = dict(lr = [args.learning_rate],
          batch_size = [args.batch_size],
          n_epoch = [args.n_epoch],
          layer_nb = [args.layer_nb],
          attr_weight = [args.attr_weight],
          dis_weight = [args.dis_weight],
          batch_norm = [args.batch_norm],
          attributes_idx = [tuple(args.attribute_indexes)],
          attributes_number = [len(args.attribute_indexes)],
          disentanglement_loss = [args.disentanglement_loss],
          knee = [args.knee]
    )
    
    start_time = time.time()
    
    train(parameters, 
          data_file = os.path.join(args.data_folder,'data.pkl'),
          label_file = os.path.join(args.data_folder,'label.pkl'),
          save_model = True,
          save_path = args.save_path)
    
    end_time = time.time()
    print("Time elapsed:", end_time-start_time)

print('Defining functions: done')
#%%

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-lr", '--learning_rate',metavar = 'Learning_rate', help = "Learning_rate", type = float, default = 2e-4)
    parser.add_argument("-bs", '--batch_size', metavar = 'batch_size', help = 'batch_size', type = int, default = 256)
    parser.add_argument("-ne",'--n_epoch', metavar = 'Number_of_epochs', help = "Number_of_epochs", type=int, default = 150)
    parser.add_argument("-ln", '--layer_nb', metavar = 'Number_of_layers', help = "Number_of_layers", type=int, default = 8)
    parser.add_argument("-aw", '--attr_weight', metavar = 'Attribute_loss_weight', help = "Weight_of_attribute_loss_term", type=float, default = 1e-5)
    parser.add_argument("-dw", '--dis_weight', metavar = 'Disentanglement_weight', help = "Weight_of_disentanglement_loss_term", type=float, default = 1e-5)
    parser.add_argument("-k", '--knee', metavar = 'knee', help = "Number of dimensions to keep after PCA", type=int, default = 60)
    parser.add_argument("-bn", '--batch_norm', metavar = 'Bath normalisation', help = "Tells if we put batch norm layers in the network", type=bool, default = True)
    parser.add_argument("-dl", '--disentanglement_loss', metavar = 'Disentanglement_loss',
                        help = "Disentangement loss to use. It can be 'Correlation' to force disentanglement or 'CorrelationPenalised' to take into account natural correlations",
                        type=str, default = 'Correlation')
    parser.add_argument("-ai", '--attribute_indexes', metavar = 'attribute_indexes', help = "attribute indexes to take into account for attribute loss and disentanglement loss.", nargs="+", type=int,  default = list(np.arange(40)))
    parser.add_argument("-df", '--data_folder', metavar = 'data_folder', help = "Folder path where to find the data files (data.pkl and label.pkl)", type=str,  default = './data/latent_attributes_dataset_gauss/')
    parser.add_argument("-sp", '--save_path', metavar = 'save_path', help = "Path where to save the model", type=str,  default = './models/disentangler/')
    args = parser.parse_args()

    if not args.disentanglement_loss in ['Correlation','CorrelationPenalised']:
        raise ValueError("-dl (--disentanglement_loss) provided must be 'Correlation' or 'CorrelationPenalised'.")
    
    main_function(args)