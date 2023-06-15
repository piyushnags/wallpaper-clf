# DL Imports
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms as T
from transformers import ViTForImageClassification

# CL Imports (avalanche-lib)
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.evaluation.metrics import (
    forgetting_metrics, accuracy_metrics,
    loss_metrics, timing_metrics, 
    confusion_matrix_metrics
)
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin, EWCPlugin
from avalanche.logging import InteractiveLogger, TextLogger

# Project Imports
from utils import get_datasets
from strategies import HFSupervised



def train():
    '''
    Description:
        Main function to train the classification model.
        Invokes necessarily loaders and sets up the CL
        environment.
    
    Args:
        None

    Returns:
        None
    '''
    # TODO: Add support for other backbone models for 
    #       comparison (get_model function)
    # Total of 17 2D Wallpaper groups
    labels = [
        'CM', 'CMM', 'P1', 'P2', 'P3', 'P3M1', 'P4', 'P4G',
        'P4M', 'P6', 'P6M', 'P31M', 'PG', 'PGG', 'PM', 'PMG', 'PMM'
    ]
    num_classes = len(labels)

    # Load the pre-trained model
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=num_classes,
        id2label = {str(i): c for i,c in enumerate(labels)},
        label2id = {c: str(i) for i,c in enumerate(labels)},
        ignore_mismatched_sizes=True,
    )

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optimizer and Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Continual Learning setup    
    # Create the class-incremental CL scenario
    train_ds, val_ds, test_ds = get_datasets()
    # NOTE: This is what the ViTImageProcessor would apply
    #       Currently, the HF-Avalanche integration for CV
    #       isn't well documented.
    train_transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    eval_transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    scenario = nc_benchmark(
        train_ds,
        val_ds,
        n_experiences=17,
        train_transform=train_transform,
        eval_transform=eval_transform,
        task_labels=False
    )

    # Evaluator plugin
    interactive_logger = InteractiveLogger()
    text_logger = TextLogger(open('logfile.txt', 'a'))
    loggers = [interactive_logger, text_logger]

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(num_classes=num_classes, save_image=False, stream=True),
        strict_checks=False,
        loggers=loggers
    )

    # Initialize the CL strategy
    replay_plugin = ReplayPlugin(mem_size=100)
    ewc_plugin = EWCPlugin(ewc_lambda=1e-3)

    cl_strategy = HFSupervised(
        model=model, 
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=128, 
        train_epochs=10,
        eval_mb_size=128,
        evaluator=eval_plugin,
        plugins=[replay_plugin, ewc_plugin],
        device=device
    )

    # Training loop
    results = []
    for experience in scenario.train_stream:
        print('Start of experience: ', experience.current_experience)
        print('Current classes: ', experience.classes_in_this_experience)

        cl_strategy.train(experience)
        results.append( cl_strategy.eval(scenario.test_stream) )
    
    print(results)



if __name__ == '__main__':
    # TODO: Add support for argparse
    train()