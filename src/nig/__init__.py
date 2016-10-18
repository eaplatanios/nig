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

__logging_config_path = os.path.join(os.path.dirname(__file__), 'logging.yaml')
if os.path.exists(__logging_config_path):
    with open(__logging_config_path, 'rt') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
else:
    logging.getLogger('').addHandler(logging.NullHandler())

from . import data
from . import evaluation
from . import learning
from . import math
from . import models
from . import utilities

__author__ = 'eaplatanios'

__all__ = ['data', 'evaluation', 'learning', 'math', 'models', 'utilities']
