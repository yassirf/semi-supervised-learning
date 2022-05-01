import pickle
import numpy as np
from typing import Dict


__all__ = [
    'UncertaintyStorage'
]


class UncertaintyStorage(Dict):
    def push(self, dinfo: Dict):
        # Push all values to storage
        for key, value in dinfo.items():
            
            # Define keys 
            if key not in self.storage: 
                self.__dict__[key] = np.array([])

            # Convert tensor to numpy array and concatenate
            self.__dict__[key] = np.concatenate((self.__dict__[key], value.cpu().detach().numpy()))

