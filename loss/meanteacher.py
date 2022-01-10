import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

from .cross_entropy import CrossEntropy

__all__ = ['crossentropy_and_meanteacher']


def softmax_mse_loss(input_logits, target_logits):
    """
    Takes softmax on both sides and returns MSE loss
    Gradients only propagated through the input logits.
    """
    input_softmax = F.softmax(input_logits, dim = -1)
    target_softmax = F.softmax(target_logits, dim = -1)
    return F.mse_loss(input_softmax, target_softmax, reduction='mean')


class MeanTeacher(CrossEntropy):
    def __init__(self, args, model, optimiser, scheduler):
        super(MeanTeacher, self).__init__(args, model, optimiser, scheduler)

        # Get loss specific arguments: mixing weight and ema parameter
        self.alpha  = args.meanteacher_alpha_ramp   # alpha used in loss
        self.alphai = args.meanteacher_alpha        # alpha initial
        self.alphaf = args.meanteacher_alpha        # alpha final

        self.w  = 0.0                           # weight used in loss
        self.wf = args.meanteacher_w            # weight final

        # Number of iterations of ramping
        self.rampi = args.meanteacher_i

        # Build an ema teacher
        self.teacher = self.build_teacher()

        # Consistency loss
        self.consistency_loss = softmax_mse_loss
    
    def get_validation_model(self):
        return self.teacher

    def build_teacher(self):
        # Get device to build model on
        device = utils.get_device(gpu = self.args.gpu)

        # Build the model using the same arguments as the student
        model = utils.loaders.load_model(self.args).to(device)

        # Detach model parameters
        for param in model.parameters():
            param.detach_()
        
        return model

    def update_teacher(self):
        # Use the true average of prior checkpoints until the exponential average is more accurate
        alpha = min(1 - 1 / (self.step + 1), self.alpha)

        # Use weighted average of student and teacher parameters
        for ema_param, param in zip(self.teacher.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def step(self, loss):
        # Update the teacher based on old student parameters
        self.update_teacher()

        # Get the ratio of ramp-up section
        ratio = self.i/self.rampi

        # Update loss specific parameters
        # Let the alpha be the initial during ramp and final after ramp
        self.alpha = self.alphai if ratio < 1.0 else self.alphaf

        # Let the mixing weight linearly increase until final value
        self.w = min(ratio, 1.0) * self.wf

        # Call the optimisation scheme and update iteration
        super(MeanTeacher, self).step(loss)

    def forward_meanteacher(self, info):
        
        # Get unlabelled images
        # For mean-teacher methods the input should have been augmented twice
        # having the shape (batch, 2, *). This is done using the composed_standard_transform
        x_ul = info['x_ul']

        # The first input is for the student model
        pred_ul, _ = self.model(x_ul[:, 0])

        with torch.no_grad():
            # The second input is for the teacher model
            teacher_ul, _ = self.teacher(x_ul[:, 1])

        # Get the loss averaged over batch
        mt = self.consistency_loss(pred_ul, teacher_ul)

        # Record metrics
        mtinfo = {'metrics': {'mt': mt.item()}}
        return mt, mtinfo

    def forward(self, info):
        # Get the cross-entropy loss and metrics
        ce, ce_info = super(MeanTeacher, self).forward(info)

        # Get the virtual adverserial training loss
        mt, _ = self.forward_meanteacher(info)

        # Compute total loss
        loss = ce + self.w * mt

        # Record metrics
        ce_info['metrics']['loss'] = loss.item()
        ce_info['metrics']['ce']   = ce.item()
        ce_info['metrics']['mt']   = mt.item()

        return loss, ce_info


def crossentropy_and_meanteacher(**kwargs):
    return MeanTeacher(**kwargs)