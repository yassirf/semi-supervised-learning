import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import sampler


class SimpleDataset(Dataset):
    def __init__(self, x, y, transform_x = None, transform_y = None):
        self.x = x
        self.y = y
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __getitem__(self, i):

        # Load input and transform
        x = self.x[i]
        if self.transform_x is not None:
            x = self.transform_x(x)
        
        # Load label and transform
        y = self.y[i]
        if self.transform_y is not None:
            y = self.transform_x(x)
        return x, y

    def __len__(self):
        return len(self.x)


class InfiniteSampler(sampler.Sampler):

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        while True:
            order = np.random.permutation(self.num_samples)
            for i in range(self.num_samples):
                yield order[i]

    def __len__(self):
        return None