# Built-in Imports
from typing import List, Tuple
import os

# DL Imports
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

# CL Imports (avalanche-lib)
from avalanche.benchmarks.utils import AvalancheDataset, as_avalanche_dataset


# TODO: Old artiact
# def generate_experiences(data_root: str = 'data/') -> Tuple[List[ Tuple[str, int] ]]:
#     '''
#     Description:
#         Function to get CL experiences from data splits (train, val, test)
#         in memory. Needed for loading data through avalanche library
    
#     Args:
#         data_root: location of dataset containing 2D wallpaper
#                    groups. Default is 'data'
        
#     Returns:
#         Tuple containing train, val, and test experiences
#     '''
#     train_prefix = os.path.join(data_root, 'train')
#     val_prefix = os.path.join(data_root, 'test')
#     test_prefix = os.path.join(data_root, 'test_challenge')

#     # Get all train experiences
#     train_experiences = []

#     for i, (root, dir, files) in enumerate(os.walk(train_prefix)):
#         if dir:
#             continue

#         curr_exp = list(map(
#             lambda x: ( os.path.join(root, x), i-1 ), files
#         ))
#         train_experiences.append(curr_exp)

#     # Get all validation experiences
#     val_experiences = []
#     for root, dir, files in os.walk(val_prefix):
#         if dir:
#             continue

#         curr_exp = list(map(
#             lambda x: ( os.path.join(root, x), i-1 ), files
#         ))
#         val_experiences.append(curr_exp)

#     # Get all test experiences
#     test_experiences = []
#     for root, dir, files in os.walk(test_prefix):
#         if dir:
#             continue

#         curr_exp = list(map(
#             lambda x: ( os.path.join(root, x), i-1 ), files
#         ))
#         test_experiences.append(curr_exp)
    
#     return train_experiences, val_experiences, test_experiences


def get_datasets(root: str = 'data') -> Tuple[AvalancheDataset]:
    '''
    Description:
        Function to get train, val, and test splits as
        torch datasets. Using the wallpaper dataset containing
        samples of 2D wallpaper groups
    
    Args:
        root: Path to data root containing the splits
    
    Returns:
        Tuple of train, val, and test splits as datasets
    '''
    # prefixes for subdirs inside data root
    train_prefix = 'train'
    val_prefix = 'test'
    test_prefix = 'test_challenge'

    # Use torchvision to create the datasets
    train_dataset =  AvalancheDataset(ImageFolder( os.path.join(root, train_prefix) ))
    val_dataset = AvalancheDataset(ImageFolder( os.path.join(root, val_prefix) ))
    test_dataset =  AvalancheDataset(ImageFolder( os.path.join(root, test_prefix) ))

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    pass