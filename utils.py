# Built-in Imports
from typing import List, Tuple
import os

# DL Imports
from torchvision.datasets import ImageFolder

# CL Imports (avalanche-lib)
from avalanche.benchmarks.utils import AvalancheDataset



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

    # Use torchvision to create the datasets and convert to TensorDataset
    train_dataset =  ImageFolder( os.path.join(root, train_prefix) )
    val_dataset = ImageFolder( os.path.join(root, val_prefix) )
    test_dataset =  ImageFolder( os.path.join(root, test_prefix) )

    return train_dataset, val_dataset, test_dataset



if __name__ == '__main__':
    pass