from __future__ import absolute_import

from .cross_entropy import crossentropy
from .distillation import crossentropy_and_distillation
from .distillation_and_proxy import crossentropy_and_distillation_and_proxy
from .vat import crossentropy_and_vat
from .meanteacher import crossentropy_and_meanteacher
from .ict import crossentropy_and_ict
from .uce import uncertainty_crossentropy
from .mcuce import sample_uncertainty_crossentropy