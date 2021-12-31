import os, errno
import numpy as np
import torch.nn as nn
import torch.optim as optim

from glob import glob
from torchvision import transforms, utils

from FOD.Loss import ScaleAndShiftInvariantLoss

def get_total_paths(path, ext):
    return glob(os.path.join(path, '*'+ext))

def get_splitted_dataset(config, split, path_images, path_depths, path_segmentation):
    list_files = [os.path.basename(im) for im in path_images]
    np.random.seed(config['General']['seed'])
    np.random.shuffle(list_files)
    if split == 'train':
        selected_files = list_files[:int(len(list_files)*config['Dataset']['splits']['split_train'])]
    elif split == 'val':
        selected_files = list_files[int(len(list_files)*config['Dataset']['splits']['split_train']):int(len(list_files)*config['Dataset']['splits']['split_train'])+int(len(list_files)*config['Dataset']['splits']['split_val'])]
    else:
        selected_files = list_files[int(len(list_files)*config['Dataset']['splits']['split_train'])+int(len(list_files)*config['Dataset']['splits']['split_val']):]

    path_images = [os.path.join(config['Dataset']['paths']['path_images'], im) for im in selected_files]
    path_depths = [os.path.join(config['Dataset']['paths']['path_depth'], im) for im in selected_files]
    path_segmentation = [os.path.join(config['Dataset']['paths']['path_segmentation'], im) for im in selected_files]
    return path_images, path_depths, path_segmentation

def get_transforms(config):
    im_size = (config['Dataset']['transforms']['resize'],config['Dataset']['transforms']['resize'])

    transform_image = transforms.Compose([
        transforms.Resize(im_size), transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    transform_depth = transforms.Compose([
        transforms.Resize(im_size), 
        transforms.ToTensor()
    ])
    transform_seg = transforms.Compose([
        transforms.Resize(im_size), 
        transforms.ToTensor()
    ])
    return transform_image, transform_depth,transform_seg

def get_loss(config):
    if config['General']['loss'] == 'mse':
        return nn.MSELoss()
    elif config['General']['loss'] == 'ssi':
        return ScaleAndShiftInvariantLoss()
    else:
        return None

def create_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def get_optimizer(config, net):
    if config['General']['optim'] == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=config['General']['lr'])
    elif config['General']['optim'] == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=config['General']['lr'], momentum=config['General']['momentum'])
    return optimizer
