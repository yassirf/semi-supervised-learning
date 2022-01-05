import os
import torch
from collections import deque

__all__ = ['Checkpointer']


class Checkpointer(object):
    def __init__(self, path, save_last_n = -1):

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

    def save_generic(self, i, acc, model, optimiser = None, filename = "checkpoint.pt"):
        path = os.path.join(self.path, filename)

        torch.save({
            'iteration': i,
            'accuracy': acc,
            'state_dict': model.state_dict(),
            'optimizer': None if optimiser is None else optimiser.state_dict(),
        }, path)

    def save_best(self, i, acc, model, optimiser = None):

        # If the accuracy is lower then do not save
        if acc <= self.acc:
            return

        # Update best accuracy found
        self.acc = acc

        # Save best model
        self.save_generic(i, acc, model, optimiser, filename = "checkpoint_best.pt")

    def save_last(self, i, acc, model, optimiser = None):
        # Save last model
        self.save_generic(i, acc, model, optimiser, filename = "checkpoint_last.pt")

    def save(self, i, acc, model, optimiser = None):
        
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
            self.save_generic(i, acc, model, optimiser, filename = self.names[-1])

        # Save last and best models
        self.save_last(i, acc, model, optimiser)
        self.save_best(i, acc, model, optimiser)