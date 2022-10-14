from typing import List

import torch
import torch.nn as nn


__all__ = [
    'Ensemble',
]


def stack_dicts(list_of_dicts, dim = 0):
    sdict = {}

    # Iterate over all dictionaries
    for d in list_of_dicts:

        # And all items
        for key, ptensor in sdict.items():

            # And append them to lists
            if key not in sdict:
                sdict[key] = []
            sdict[key].append(ptensor)
    
    # Stack all lists within the result
    for key, value in sdict.items():
        sdict[key] = torch.stack(value, dim = dim)
    
    return sdict


class Ensemble(nn.Module):
    def __init__(self, models: List):
        super(Ensemble, self).__init__()

        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor):

        # Produce the outputs from each model
        outputs = [m(x) for m in self.models]

        # Process: Stack model predictions and average
        out = torch.stack([op[0] for op in outputs], dim = 0)
        out = torch.log_softmax(out, dim = -1)
        out = torch.logsumexp(out, dim = 0) - math.log(out.size(0))

        # Stack all the dictionaries
        info = stack_dicts([op[1] for op in outputs], dim = 0)

        return out, info
