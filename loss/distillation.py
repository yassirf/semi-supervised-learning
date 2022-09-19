import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

from .base import accuracy
from .cross_entropy import CrossEntropy

import logging 
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

__all__ = ['crossentropy_and_distillation']


def kl_divergence_loss(input_logits, target_logits, temperature):
    """
    Computes the temperature annealed kl-divergence
    Gradients only propagated through the input logits.
    """
    input_lsoftmax = F.log_softmax(input_logits/temperature, dim = -1)
    target_lsoftmax = F.log_softmax(target_logits/temperature, dim = -1)

    loss = torch.exp(target_lsoftmax) * (target_lsoftmax - input_lsoftmax)
    loss = (loss.sum(-1)).mean()
    return loss


class Distillation(CrossEntropy):
    def __init__(self, args, model, optimiser, scheduler):
        super(Distillation, self).__init__(args, model, optimiser, scheduler)

        # Get loss specific arguments for knowledge distillation
        self.distillation_w  = args.teacher_ratio         # weight used in loss
        self.distillation_t  = args.temperature           # temperature used in loss

        # Build an ema teacher
        self.teacher = self.build_teacher()

        # Consistency loss
        self.consistency_loss = kl_divergence_loss

    def get_validation_model(self):
        return self.model

    def build_teacher(self):
        # Get device to build model on
        device = utils.get_device(gpu = self.args.gpu)

        # Log the teacher information
        logger.info("Building teacher; architecture = {}".format(self.args.teacher_arch))
        logger.info("Building teacher; path = {}".format(self.args.teacher_path))

        # Build the model using the same arguments as the student
        model = utils.loaders.load_model(self.args, attribute = "teacher_arch").to(device)
        
        # Checkpointer for loading the best checkpoint, the path points to base directory
        checkpointer = utils.Checkpointer(self.args, path = self.args.teacher_path, save_last_n = -1)
        model.load_state_dict(checkpointer.load(device)['state_dict'])

        # Detach model parameters
        for param in model.parameters():
            param.detach_()
        
        return model

    def forward(self, info):

        # Get labelled image and label
        x_l, y_l = info['x_l'], info['y_l']

        # Perform model forward pass
        pred_l, _ = self.model(x_l)

        with torch.no_grad():
            # The second input is for the teacher model
            teacher_l, _ = self.teacher(x_l)

        # Compute ce-loss
        ce = self.ce(pred_l, y_l)

        # Get the kl-loss averaged over batch
        kd = self.consistency_loss(pred_l, teacher_l, self.distillation_t)

        # Compute total loss
        loss = (1 - self.distillation_w) * ce + self.distillation_w * kd * self.distillation_t ** 2

        # Compute accuracy
        acc = accuracy(pred_l.detach().clone(), y_l, top_k = (1, 5))

        # Record metrics
        linfo = {'metrics': {
            'loss': loss.item(),
            'ce': ce.item(),
            'kd': kd.item(),
            'acc1': acc[0].item(),
            'acc5': acc[1].item(),
        }}

        return loss, linfo


def crossentropy_and_distillation(**kwargs):
    return Distillation(**kwargs)