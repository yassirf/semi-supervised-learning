import torch

__all__ = ['sgd', 'nesterovsgd', 'rmsprop']


def sgd(args, model):
    return torch.optim.SGD(
        model.parameters(),
        lr = args.learning_rate,
        momentum = args.momentum,
        weight_decay = args.weight_decay,
    )


def nesterovsgd(args, model):
    return torch.optim.SGD(
        model.parameters(),
        lr = args.learning_rate,
        momentum = args.momentum,
        weight_decay = args.weight_decay,
        nesterov  =True
    )


def rmsprop(args, model):
    return torch.optim.RMSprop(
        model.parameters(),
        lr = args.learning_rate,
        momentum = args.momentum,            
        weight_decay = args.weight_decay,
        eps = 0.0316,
        alpha = 0.9,
    )