import uncertainty as lib

__all__ = ['load_uncertainty']


# Get all uncertainty calculators available in the folder given they are lower case and callable 
names = sorted(name for name in lib.__dict__ if name.islower() and callable(lib.__dict__[name]))


# Define model loader
def load_uncertainty(args):
    """
    General uncertainty loader
    """
    if args.uncertainty_name not in names:
        raise ValueError("The value {} is not one of the valid choices: {}".format(args.uncertainty_name, names))
    
    return lib.__dict__[args.uncertainty_name]()