import loss as lib

__all__ = ['load_loss']


# Get all cifar models available in the folder given they are lower case and callable 
names = sorted(name for name in lib.__dict__ if name.islower() and callable(lib.__dict__[name]))

# Define model loader
def load_loss(args, model, optimiser, scheduler):
    """
    General model loader
    """
    if args.loss not in names:
        raise ValueError("The value {} is not one of the valid choices: {}".format(args.loss, names))
    
    loss = lib.__dict__[args.loss]
    loss = loss(
        args = args, 
        model = model, 
        optimiser = optimiser, 
        scheduler = scheduler,
    )
    return loss