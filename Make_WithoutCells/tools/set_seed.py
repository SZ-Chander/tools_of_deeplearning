# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 23:56:07 2021

@author: admin
"""

import random
import torch
import numpy as np

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

