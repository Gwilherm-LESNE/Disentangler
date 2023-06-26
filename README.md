# A Compact and Semantic Latent Space for Disentangled and Controllable Image Editing

Official implementation of the paper *A Compact and Semantic Latent Space for Disentangled and Controllable Image Editing*

![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)
![pytorch 1.12.1](https://img.shields.io/badge/Pytorch-1.12.0-blue.svg)
![Cuda 11.6](https://img.shields.io/badge/Cuda-11.6-yellow.svg)

![image](./data/figure.png)
**Figure:** *Several edits made by our method*

> **A Compact and Semantic Latent Space for Disentangled and Controllable Image Editing** <br>
>  Gwilherm Lesné, Yann Gousseau, Saïd Ladjal, Alasdair Newson <br>
>  LTCI, Télécom Paris <br>

## Set up

Clone this repository
```bash
git clone https://github.com/Gwilherm-LESNE/Disentangler.git
cd Disentangler/
```

Install the required libraries
```bash
conda env create -f disentangler.yml
```

Load the conda environment
```bash
conda activate disentangler
```
**N.B.** This code relies on the official pytorch StyleGAN2-Ada's implementation. Please follow the **requirements** [here](https://github.com/NVlabs/stylegan2-ada-pytorch).

## Getting data

You have two options: 
- You want to use the data we used (and hence, the same StyleGAN2 generator):
  - Download the files [here](https://drive.google.com/drive/folders/1MJbEHwa0sYolDI4W3vYUmv-j8fXqak_u?usp=sharing)
  - Put `stylegan2.pkl` in `models` folder
  - Unzip `latent_attributes_dataset_gauss.zip` in `data` folder
- You want to train the model on your own data:
  - Get a pretrained StyleGAN2 generator in the same format as the [official](https://github.com/NVlabs/stylegan2-ada-pytorch) implementation, name it `stylegan2.pkl` and save it in `models` folder
  - Create with your StyleGAN2 model and a pretrained classifier your database: `data.pkl` which contains the $\mathcal{W}$ latent vectors (torch.tensor of format (Nb_of_data, 512)) and `label.pkl` which contains the corresponding attributes (torch.tensor of format (Nb_of_data, Nb_of_attributes))
  - Store these files in a folder, for example in `./data/path_to_dataset/`
  - Gaussianize this dataset using the `dataset.py` file:
    ```
    python dataset.py ./data/path_to_dataset -s ./data/path_to_save/
    ```

## Training

```bash
python train.py
```
Options:
  - `-lr` Learning rate
  - `-ln` Number of dense layers to use for both your encoder and decoder.
  - `-bs` Batch size
  - `-ne` Number of epochs
  - `-bn` Tells if you put batch normalisation layers in your auto-encoder
  - `-aw` Weight for attribute loss term
  - `-dw` Weight for disentanglement loss term
  - `-k` Number of PCA dimensions to keep
  - `-dl` Disentanglement loss, you may want to disentangle the latent space or take into account the natural correlations of your data. The default is the former. For the latter, the CelebA attribute correlations will be used.
  - `-ai` Indices of the attributes you want to take into account
  - `-df` Path to the folder where the data is stored (`data.pkl` and `label.pkl` files)
  - `-sp` Path to the folder you want to save the model in

To visualize your training:
```bash
tensorboard --logdir=models
```

## Running a pretrained model

```bash
python edit.py -c code_value -a attr_index
```
Options:
  - `-c` Code value. We recommend to put 2.5 or -2.5 depending on wether you want to add or remove the attribute.
  - `-a` Attribute index. Indicates which attribute to edit based on its index.
  - `-s` Seed, used for sampling in $\mathcal{W}$
  - `-sm` sample mode. `0` is sampling in the dataset you used to train. `1` is random sampling in $\mathcal{W}$.
  - `-nf` Path to the pretrained network file
  - `-ln` Number of layers in the network
  - `-k` Number of PCA dimensions kept by the network
  - `-bn` Indicates if the network has batch norm layers or not
  - `-df` Path to the dataset. Used if `-sm` is 0

## Licence

All rights reserved. The code is released for academic research use only.

## Citation

If you use our code/data, please cite our paper.

## Acknowledgments

This work is built upon the one done by Karras et al. You can find it [here](https://github.com/NVlabs/stylegan2-ada-pytorch). 
