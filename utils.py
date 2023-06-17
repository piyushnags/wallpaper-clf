# Built-in Imports
from typing import List, Tuple, Any
import os
import argparse

# DL Imports
import torch
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification

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


def get_model(arch: str = 'vit'):
    '''
    Description:
        Helper function to get model based on 
        architecture specification.
    
    Args:
        arch: architecture type (str)
    
    Returns:
        model: nn.Module 
    '''
    if arch == 'vit':
        # Labels for all 17 2D Wallpaper groups
        labels = [
            'CM', 'CMM', 'P1', 'P2', 'P3', 'P3M1', 'P4', 'P4G',
            'P4M', 'P6', 'P6M', 'P31M', 'PG', 'PGG', 'PM', 'PMG', 'PMM'
        ]
        num_classes = len(labels)
        
        # Load pretrained ViT model
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=num_classes,
            id2label = {str(i): c for i,c in enumerate(labels)},
            label2id = {c: str(i) for i,c in enumerate(labels)},
            ignore_mismatched_sizes=True,
        )

        # Freeze ViT backbone
        for name, param in model.named_parameters():
            if 'encoder' in name:
                param.requires_grad = False
        
        return model
    
    else:
        raise ValueError(f'Model architecture {arch} is invalid')


def get_device(device: str):
    '''
    Description:
        Function to get torch device based on
        specification and availability
    
    Args:
        device: string input spec
    
    Returns:
        available torch device
    '''
    if device == 'cuda':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device('cpu')


def parse() -> Any:
    '''
    Description:
        Get parameters for training pipeline and return
        args Namespace
    
    Args:
        None
    
    Returns:
        args: Namespace
    '''
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Model and device spec
    parser.add_argument('--device', type=str, default='cuda', help='Device to be used for training')
    parser.add_argument('--arch', type=str, default='vit', help='model architecture for training')
    
    # Standard DL training params
    parser.add_argument('--batch_size', type=int, default=128, help='Mini-batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay for AdamW Optimizer')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training per experience')

    # CL training params
    parser.add_argument('--ewc_lambda', type=float, default=1e-3, help='Lambda to be used when using EWC strategy')
    parser.add_argument('--replay_buf_size', type=int, default=200, help='Size of Replay buffer when using Replay strategy')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    pass