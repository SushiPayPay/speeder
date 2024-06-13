from copy import deepcopy
import os
import random
import numpy as np
import torch

class DotDict(dict):
    def __getitem__(self, item):
        value = dict.__getitem__(self, item)
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        return value

    __getattr__ = __getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo):
        return DotDict(deepcopy(dict(self), memo))

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
    
sty = DotDict({
        'BLUE': '\033[0;34m',
        'CYAN': '\033[0;36m',
        'RED': '\033[0;31m',
        'PINK': '\033[0;35m',
        'GREEN': '\033[0;32m',
        'BOLD': '\033[1m',
        'ITALIC': '\033[3m',
        'RESET': '\033[0m'
        })

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    torch.manual_seed(seed)

def set_gpus(ids):
    '''Takes in a list of integers or a single integer representing GPU IDs to expose to the python process
    Passing -1 will expose all available GPUs
    '''
    gpus = []
    try:
        if isinstance(ids, int):
            gpus = [str(i) for i in range(torch.cuda.device_count())] if ids == -1 else [str(ids)]
        else:
            gpus = [str(i) for i in ids]
    except:
        raise ValueError("<ids> must be an int or container of ints")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    return gpus
