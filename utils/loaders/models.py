import models.cifar as cifar

__all__ = ['load_model']


# Get all cifar models available in the folder given they are lower case and callable 
names = sorted(name for name in cifar.__dict__ if name.islower() and callable(cifar.__dict__[name]))


# Define model loader
def load_model(args):
    """
    General model loader
    """
    if args.arch not in names:
        raise ValueError("The value {} is not one of the valid choices: {}".format(args.arch, names))
    
    model = cifar.__dict__[args.arch]
    model = model(
        num_classes = args.num_classes      
        # Add additional arguments to the model here if needed
    )
    return model