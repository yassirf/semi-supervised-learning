import optim as lib

__all__ = ['load_optim']


# Get all cifar models available in the folder given they are lower case and callable 
names = sorted(name for name in lib.__dict__ if name.islower() and callable(lib.__dict__[name]))


# Define model loader
def load_optim(args, model):
    """
    General optimiser loader
    """
    if args.optim not in names:
        raise ValueError("The value {} is not one of the valid choices: {}".format(args.optim, names))

    return lib.__dict__[args.optim](args, model)