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

from . import array_ops
from . import variable_ops
from . import weights_broadcast_ops

from .array_ops import *
from .variable_ops import *
from .weights_broadcast_ops import *

__author__ = 'eaplatanios'

__all__ = ['array_ops', 'variable_ops', 'weights_broadcast_ops']
__all__.extend(array_ops.__all__)
__all__.extend(variable_ops.__all__)
__all__.extend(weights_broadcast_ops.__all__)
