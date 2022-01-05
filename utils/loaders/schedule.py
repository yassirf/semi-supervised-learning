import schedule as lib

__all__ = ['load_schedule']


# Get all cifar models available in the folder given they are lower case and callable 
names = sorted(name for name in lib.__dict__ if name.islower() and callable(lib.__dict__[name]))


# Define model loader
def load_schedule(args, optim):
    """
    General schedule loader
    """
    if args.lr_scheduler not in names:
        raise ValueError("The value {} is not one of the valid choices: {}".format(args.lr_scheduler, names))
    
    return lib.__dict__[args.lr_scheduler](args, optim)