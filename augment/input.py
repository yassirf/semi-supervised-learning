import torch
import torchvision.transforms as transforms

__all__ = [
    'none_transform',
    'standard_transform',
    'composed_standard_transform',
]

def none_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def standard_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def composed_standard_transform(num = 2):

    # Get the standard transformer
    transform = standard_transform()

    # Define a stochastic augmenter which stacks images
    def repeated(x, num = num):
        return torch.stack([transform(x) for _ in range(num)])
    
    # Return the augmentation with stacking mechanism 
    return repeated