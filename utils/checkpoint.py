import os
import torch
from collections import deque

__all__ = ['Checkpointer']


class Checkpointer(object):
    def __init__(self, args, path, save_last_n = -1):
        self.args = args

        # Path to saving directory 
        self.path = path
        os.makedirs(path, exist_ok = True)

        # Number of checkpoints to save (default to save only the best and last)
        self.save_last_n = save_last_n
        self.save_interm = save_last_n > 0

        # Number of saved models
        self.n = 0

        # Name of models saved
        self.names = deque()

        # Best accuracy so far:
        self.acc = 0.0

    def save_generic(self, i, acc, model, loss = None, optimiser = None, filename = "checkpoint.pt"):
        path = os.path.join(self.path, filename)

        torch.save({
            'iteration': i,
            'accuracy': acc,
            'state_dict': model.state_dict(),
            'optimizer': None if optimiser is None else optimiser.state_dict(),
            'loss': loss,
        }, path)

    def save_best(self, i, acc, model, loss = None, optimiser = None):

        # If the accuracy is lower then do not save
        if acc <= self.acc:
            return

        # Update best accuracy found
        self.acc = acc

        # Save best model
        self.save_generic(i, acc, model, loss, optimiser, filename = "checkpoint_best.pt")

    def save_last(self, i, acc, model, loss = None, optimiser = None):
        # Save last model
        self.save_generic(i, acc, model, loss, optimiser, filename = "checkpoint_last.pt")

    def save(self, i, acc, model, loss = None, optimiser = None):
        
        # First check if we should checkpoint
        if self.save_interm:

            # Now delete last saved model if exceeding limit
            if len(self.names) >= self.save_last_n:

                # Get last saved model and delete
                os.remove(
                    os.path.join(self.path, self.names.popleft())
                )
            
            # Now move on and save next model
            self.n += 1

            # Name of model
            self.names.append("checkpoint{}.pt".format(self.n))

            # Save model
            self.save_generic(i, acc, model, loss, optimiser, filename = self.names[-1])

        # Save last and best models
        self.save_last(i, acc, model, loss, optimiser)
        self.save_best(i, acc, model, loss, optimiser)
    
    def load(self, filename = "checkpoint_best.pt"):
        # For when the argument is None
        path = self.args.load_path or os.path.join(self.path, filename)

        # Load checkpoint based on filename
        ckpt = torch.load(path)

        # Update parameters
        self.acc = ckpt['accuracy']

        return ckpt