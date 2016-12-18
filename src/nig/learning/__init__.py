# Copyright 2016, The NIG Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import absolute_import, division, print_function

import nig.learning.callbacks as callbacks
import nig.learning.learners as learners
import nig.learning.metrics as metrics
import nig.learning.models as models
import nig.learning.optimizers as optimizers

from nig.learning.callbacks import *
from nig.learning.learners import *
from nig.learning.metrics import *
from nig.learning.models import *
from nig.learning.optimizers import *

__author__ = 'eaplatanios'

__all__ = ['callbacks', 'learners', 'metrics', 'models', 'optimizers']
__all__.extend(callbacks.__all__)
__all__.extend(learners.__all__)
__all__.extend(metrics.__all__)
__all__.extend(models.__all__)
__all__.extend(optimizers.__all__)
