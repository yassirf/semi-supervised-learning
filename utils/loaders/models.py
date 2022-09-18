import models.cifar as cifar
from .counts import load_counts

__all__ = ['load_model']


# Get all cifar models available in the folder given they are lower case and callable 
names = sorted(name for name in cifar.__dict__ if name.islower() and callable(cifar.__dict__[name]))


# Define model loader
def load_model(args, attribute = "arch"):
    """
    General model loader
    """

    # Extract model architecture
    arch = getattr(args, attribute, None)

    # Ensure model has been defined and satisfies convention
    if arch not in names:
        raise ValueError("The value {} is not one of the valid choices: {}".format(arch, names))

    model = cifar.__dict__[arch]
    model = model(
        # Base required argument for all models
        num_classes = args.num_classes,

        ### Add additional arguments to the model here if needed
        # Posterior Network parameters
        latent_dim = args.latent_dim,
        flow_length = args.flow_length,
        counts = load_counts(args),

        # Monte-carlo based UCE training
        num_samples = args.uce_num_samples,
    )
    return model