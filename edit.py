#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings 
warnings.filterwarnings("ignore")

import torch
import class_utils as cu
import pickle
import dnnlib
import torch_utils
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

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

class Editor():
    def __init__(self, network, stylegan_path = './models/ffhq.pkl'):
        
        self.net = network
        self.stylegan_path = stylegan_path
        
    def edit(self, w, attr_idx, code_value):
        
        if (attr_idx.__class__,code_value.__class__) not in [(int,float),(list,list)]:
            raise ValueError('attr_idx and code_value arguments must be respectively either int & float, either list of int and list of floats.')
        if (attr_idx.__class__,code_value.__class__) == (list,list) and len(attr_idx) != len(code_value):
            raise ValueError("attr_idx and code_value are lists but don't have the same length, please fix it.")
        if (attr_idx.__class__,code_value.__class__) == (list,list) and len(attr_idx) != len(w):
            raise ValueError("w first dim (bsize) doesn't have the same length as code_value and attr_idx, please fix it.")

        c = self.net.encode(w.cuda())
                
        c_neg = c.clone().cuda()
        c_pos = c.clone().cuda()
        
        if code_value.__class__ == float:
            code_value = torch.tensor([code_value])
        if attr_idx.__class__ == int:
            attr_idx = torch.tensor([attr_idx])
            
        for i in range(w.shape[0]):
            c_pos[i,attr_idx[i]] = code_value[i]
        
        w_neg = self.net.decode(c_neg, w)
        w_pos = self.net.decode(c_pos, w)
        
        for i in range(w.shape[0]):
           with open(self.stylegan_path,'rb') as f: #Import the model
               G = pickle.load(f)['G_ema']
               
           out_neg = G.synthesis(w2Wp(w_neg[i]).detach().cpu(), noise_mode='const', force_fp32=True)
           img_neg = tensor2Image(out_neg.detach().cpu())[0]
                
           out_pos = G.synthesis(w2Wp(w_pos[i]).detach().cpu(), noise_mode='const', force_fp32=True)
           img_pos = tensor2Image(out_pos.detach().cpu())[0]
    
           plt.figure()
           image = np.concatenate((img_neg,img_pos),axis=1)
           plt.imshow(image)
           plt.title('Original image (left) and edited one (right)')
           plt.xticks(ticks=[], labels=[])
           plt.yticks(ticks=[], labels=[])
           plt.show()
           
        return  None
            
class WSampler():
    def __init__(self, LADataset_path = './data/latent_attributes_dataset_gauss', stylegan_path = './models/ffhq.pkl'):
        super(WSampler, self).__init__()
        
        ladataset = cu.LADataset(data_file = os.path.join(LADataset_path,'data.pkl'), label_file = os.path.join(LADataset_path,'label.pkl'))
        
        self.attributes = (ladataset).labels
        self.ws = (ladataset).data
        
        self.stylegan_path = stylegan_path
    
    def sample(self, n = 1, attr_idx = None, code_value = None, seed=None):
        
        if seed is not None:
            torch.manual_seed(seed)
           
        if (attr_idx is not None) and (code_value is not None):
            
            if attr_idx.__class__ == int:
                ws_kept = self.ws
                ### Keep only the indices of latent codes where attributes are more extreme than the value desired (code_value)
                indices = torch.where(self.attributes[:,attr_idx]*np.sign(self.attributes[:,attr_idx]) >= code_value*np.sign(code_value))[0]
                ### Be aware of the sign of the code_value desired
                indices = indices[torch.where(np.sign(self.attributes[indices,attr_idx])==np.sign(code_value))[0]]
                
                ws_kept = ws_kept[indices,:]
            elif attr_idx.__class__ == list:
                ws_kept = [self.ws]*len(attr_idx)
                for i in range(len(attr_idx)):
                    ### Keep only the indices of latent codes where attributes are more extreme than the value desired (code_value)
                    tmp_indices = torch.where(self.attributes[:,attr_idx[i]]*np.sign(self.attributes[:,attr_idx[i]]) >= code_value[i]*np.sign(code_value[i]))[0]
                    ### Be aware of the sign of the code_value desired
                    tmp_indices = tmp_indices[torch.where(np.sign(self.attributes[tmp_indices,attr_idx[i]])==np.sign(code_value[i]))[0]]                
                    ws_kept[i] = ws_kept[i][tmp_indices,:]
            else:
                raise ValueError('attr_idx has wrong type, must be int or list of ints')
        elif (attr_idx is None) or (code_value is None):
            
            with open(self.stylegan_path,'rb') as f: #Import the model
                G = pickle.load(f)['G']
            z = torch.randn(n,512)
            w = G.mapping(z,None,truncation_psi=0.7)[:,0,:]
            return w
            
        else:
            raise ValueError('Wrong value or type for attr_idx or code_value, please check them')
         
        if attr_idx.__class__ == list:
            ws_out = torch.zeros(len(attr_idx),self.ws.shape[1])
            for i in range(len(attr_idx)):
                idx = torch.randint(len(ws_kept[i]),(1,))
                ws_out[i] = ws_kept[i][idx].float()
            return ws_out                
        else:
            idx = torch.randint(len(ws_kept),(n,))
            return ws_kept[idx].float()


def execute(args):
    # --- Load network --- #
    net = cu.Disentangler(layer_nb = args.layer_nb, knee = args.knee, batch_norm = args.batch_norm)
    net.load_state_dict(torch.load(args.network_file))
    net.Mapin.cuda()
    net.Mapout.cuda()
    net.eval()
        
    # --- Sample w latent vector --- #
    wS = WSampler(LADataset_path = args.dataset_folder)
    if args.sample_mode == 0:
        
        if (args.attr_idx.__class__,args.code_value.__class__) not in [(int,float),(list,list)]:
            raise ValueError('-a and -c arguments must be respectively either int & float, either list of int and list of floats.')
        if (args.attr_idx.__class__,args.code_value.__class__) == (list,list) and len(args.attr_idx) != len(args.code_value):
            raise ValueError("-a and -c arguments are lists but don't have the same length, please fix it.")
        
        if args.attr_idx.__class__ == int:
            w = wS.sample(n = len(args.code_value), attr_idx=args.attr_idx, code_value=-np.sign(args.code_value), seed=args.seed)
        else:
            w = wS.sample(n = len(args.code_value), attr_idx=args.attr_idx, code_value=[-np.sign(el) for el in args.code_value], seed=args.seed)

    elif args.sample_mode == 1:
            w = wS.sample(n = len(args.code_value), seed=args.seed)
    else:
        raise ValueError(f"-sm argument must be 0 or 1, you've provided {args.sample_mode}")
    
    # --- Apply edit --- #
    editor = Editor(net)
    editor.edit(w.cuda(), args.attr_idx, args.code_value)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-c", '--code_value',metavar = 'Code_value', help = "Code_value", nargs="+", type = float, default = None)
    parser.add_argument("-a", '--attr_idx', metavar = 'Attribute_index', help = 'Attribute to be modified', nargs="+", type = int, default = None)
    parser.add_argument("-s",'--seed', metavar = 'Seed', help = "Seed", type=int, default = 0)
    parser.add_argument("-sm", '--sample_mode', metavar = 'Sample_mode', help = "Sample_mode, 0 is conditionnal sampling while 1 is random W sampling", type=int, default = 0)
    parser.add_argument("-nf", '--network_file', metavar = 'Network_path', help = "Path to the network file", type=str, default = './models/disentangler.pt')
    parser.add_argument("-ln", '--layer_nb', metavar = 'Layer number', help = "Number of layers of your network", type=int, default = 8)
    parser.add_argument("-k", '--knee', metavar = 'knee', help = "Number of dimensions to keep after PCA", type=int, default = 60)
    parser.add_argument("-bn", '--batch_norm', metavar = 'Bath normalisation', help = "Tells if we put batch norm layers in the network", type=bool, default = True)
    parser.add_argument("-df", '--dataset_folder', metavar = 'Dataset_folder', help = "Path to the dataset", type=str, default = './data/latent_attributes_dataset_gauss')
    args = parser.parse_args()

    if args.code_value is None or args.attr_idx is None:
        raise ValueError("-c or -a argument is None. You MUST provide both of them. they must be respectively float and int OR lists of these type having the same length.")
    
    execute(args)
    

