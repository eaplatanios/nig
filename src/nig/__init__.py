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

import logging.config
import os
import yaml

import nig.data as data
import nig.evaluation as evaluation
import nig.learning as learning
import nig.math as math
import nig.models as models
import nig.ops as ops
import nig.plotting as plotting
import nig.utilities as utilities

from nig.data import *
from nig.evaluation import *
from nig.learning import *
from nig.math import *
from nig.models import *
from nig.ops import *
from nig.plotting import *
from nig.utilities import *

__author__ = 'eaplatanios'

__all__ = ['data', 'evaluation', 'learning', 'math', 'models', 'ops',
           'plotting', 'utilities']
__all__.extend(data.__all__)
__all__.extend(evaluation.__all__)
__all__.extend(learning.__all__)
__all__.extend(math.__all__)
__all__.extend(models.__all__)
__all__.extend(ops.__all__)
__all__.extend(plotting.__all__)
__all__.extend(utilities.__all__)

__logging_config_path = os.path.join(os.path.dirname(__file__), 'logging.yaml')
if os.path.exists(__logging_config_path):
    with open(__logging_config_path, 'rt') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
else:
    logging.getLogger('').addHandler(logging.NullHandler())
