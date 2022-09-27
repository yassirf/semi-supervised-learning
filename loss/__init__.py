from __future__ import absolute_import

from .cross_entropy import crossentropy
from .vat import crossentropy_and_vat
from .meanteacher import crossentropy_and_meanteacher
from .ict import crossentropy_and_ict
from .uce import uncertainty_crossentropy
from .mcuce import sample_uncertainty_crossentropy

from .distillation import *