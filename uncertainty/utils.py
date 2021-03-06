import pickle
import numpy as np
from typing import Dict


__all__ = [
    'UncertaintyStorage'
]

class UncertaintyStorage(dict):
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        if key not in self: return []
        val = dict.__getitem__(self, key)
        return val

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v.detach().cpu().tolist()
            
    def uconcatenate(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] += v.detach().cpu().tolist()