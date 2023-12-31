# Built-in Imports
from typing import Any

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
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.logging import InteractiveLogger, TextLogger

# Project Imports
from utils import get_datasets, get_model, get_device, parse
from strategies import HFSupervised
from plugins import HFEWCPlugin



def train(args: Any):
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
    model = get_model(args.arch)

    # Move model to device
    device = get_device(args.device)
    model.to(device)

    # Optimizer and Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
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
        confusion_matrix_metrics(num_classes=17, save_image=False, stream=True),
        strict_checks=False,
        loggers=loggers
    )

    # Initialize the CL strategy
    replay_plugin = ReplayPlugin(mem_size=args.replay_buf_size)
    ewc_plugin = HFEWCPlugin(ewc_lambda=args.ewc_lambda)

    cl_strategy = HFSupervised(
        model=model, 
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=args.batch_size, 
        train_epochs=args.num_epochs,
        eval_mb_size=args.batch_size,
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
    args = parse()
    train(args)