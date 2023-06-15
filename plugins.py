# Built-in Imports
from typing import Dict, Union
import warnings

# DL Imports
import torch
from torch.utils.data import DataLoader

# CL Imports (avalanche-lib)
from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins import EWCPlugin
from avalanche.training.utils import zerolike_params_dict, ParamData 



class HFEWCPlugin(EWCPlugin):
    '''
    Updated class that integrates HuggingFace
    model with the EWC plugin. It's the same
    as the original class except that the loss computation
    is modified.
    '''
    def __init__(
        self,
        ewc_lambda,
        mode='separate',
        decay_factor=None,
        keep_importance_data=False
    ):
        super(HFEWCPlugin, self).__init__(
            ewc_lambda, 
            mode, 
            decay_factor, 
            keep_importance_data
        )
    

    def compute_importances(
            self, model, criterion, optimizer, dataset, device, batch_size
    ) -> Dict[str, ParamData]:
        
        model.eval()

        # Set RNN-like modules on GPU to training mode to avoid CUDA error
        if device == "cuda":
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        "RNN-like modules do not support "
                        "backward calls while in `eval` mode on CUDA "
                        "devices. Setting all `RNNBase` modules to "
                        "`train` mode. May produce inconsistent "
                        "output if such modules have `dropout` > 0."
                    )
                    module.train()

        # list of list
        importances = zerolike_params_dict(model)
        collate_fn = (
            dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn
        )

        for i, batch in enumerate(dataloader):
            # get only input, target and task_id from the batch
            x, y, task_labels = batch[0], batch[1], batch[-1]
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # Modified this part to output tensors from
            # HF model (single-task only)
            # out = avalanche_forward(model, x, task_labels)
            inputs = {
                'pixel_values' : x,
                'labels' : y
            }
            out = model(**inputs).logits
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                model.named_parameters(), importances.items()
            ):
                assert k1 == k2
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))

        model.train()

        return importances