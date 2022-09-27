from __future__ import absolute_import

from .distillation import crossentropy_and_distillation
from .unlabelled_distillation import crossentropy_and_unlabelled_distillation

from .distillation_and_proxy import crossentropy_and_distillation_and_proxy
from .distillation_and_proxy import crossentropy_and_distillation_and_proxy_entropy_mse

from .unlabelled_distillation_and_proxy import unlabelled_distillation_and_proxy_only
from .unlabelled_distillation_and_proxy import unlabelled_distillation_and_proxy_entropy_mse_only