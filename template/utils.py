from copy import deepcopy

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