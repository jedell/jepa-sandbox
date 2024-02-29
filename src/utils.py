import logging
import sys

import torch
from torch.nn import Module

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def init_model() -> (Module, Module):

    # init modules, model params
    # encoder: ViT
    # predictor: ViT 
    
    # init weights

    # send to device

    return