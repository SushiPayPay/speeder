from copy import deepcopy
import os
import random
import numpy as np
import torch
import lightning as L

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

class sty:
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    RED = '\033[0;31m'
    PINK = '\033[0;35m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[0;33m'
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    RESET = '\033[0m'

    BG_RED = '\033[0;41m'
    BG_GREEN = '\033[0;42m'
    BG_YELLOW = '\033[0;43m'
    BG_BLUE = '\033[0;44m'
    BG_MAGENTA = '\033[0;45m'
    BG_CYAN = '\033[0;46m'
    BG_WHITE = '\033[0;47m'
    BG_GREY = "\033[0;100m"

def color_print(*args, color=sty.RESET, **kwargs):
    print(color, end='')
    print(*args, **kwargs, end='')
    print(sty.RESET)

def pl(*args, prefix='[INFO]', **kwargs):
    print(f'{sty.BG_GREY}{prefix}', end=' ')
    print(*args, **kwargs, end='')
    print(sty.RESET)

def pb(*args, **kwargs): color_print(*args, color=sty.BLUE, **kwargs)
def pc(*args, **kwargs): color_print(*args, color=sty.CYAN, **kwargs)
def pr(*args, **kwargs): color_print(*args, color=sty.RED, **kwargs)
def pp(*args, **kwargs): color_print(*args, color=sty.PINK, **kwargs)
def pg(*args, **kwargs): color_print(*args, color=sty.GREEN, **kwargs)
def py(*args, **kwargs): color_print(*args, color=sty.YELLOW, **kwargs)

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    torch.manual_seed(seed)
    L.seed_everything(seed)

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
    gpus = [int(gpu) for gpu in gpus]

    return gpus
