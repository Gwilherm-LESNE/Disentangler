# A Compact and Semantic Latent Space for Disentangled and Controllable Image Editing

Official implementation of the paper *A Compact and Semantic Latent Space for Disentangled and Controllable Image Editing*

![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)
![pytorch 1.12.1](https://img.shields.io/badge/Pytorch-1.12.0-blue.svg)
![Cuda 11.6](https://img.shields.io/badge/Cuda-11.6-yellow.svg)

## Set up

Clone this repository.
```bash
git clone https://github.com/Gwilherm-LESNE/Disentangler.git
cd Disentangler/
```

Install the required libraries
```bash
conda env create -f disentangler.yml
```
**N.B.** This code relies on the official pytorch StyleGAN2-Ada's implementation. Please follow the **Requirements** [here](https://github.com/NVlabs/stylegan2-ada-pytorch).

## Getting data

You have two options: 
- You want to use the data we used (and hence, the same StyleGAN2 generator):
  - Download the files [here](https://drive.google.com/drive/folders/1MJbEHwa0sYolDI4W3vYUmv-j8fXqak_u?usp=sharing)
  - Put *stylegan2.pkl* in *models* folder
  - Unzip *latent_attributes_dataset_gauss.zip* in *data* folder
- You want to train the model on your own data
  - Get a pretrained StyleGAN2 generator in the same format as the official implementation and 
  - Step two

## Training

```bash
python train.py
```
Options:


## Running a pretrained model

```bash
python edit.py
```
