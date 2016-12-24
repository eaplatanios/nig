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

import nig.data.loaders as loaders
import nig.data.converters as converters
import nig.data.filters as filters
import nig.data.iterators as iterators
import nig.data.processors as processors
import nig.data.utilities as utilities

from nig.data.converters import *
from nig.data.filters import *
from nig.data.iterators import *
from nig.data.processors import *
from nig.data.utilities import *

__author__ = 'eaplatanios'

__all__ = ['loaders', 'converters', 'filters', 'iterators', 'processors',
           'utilities']
__all__.extend(converters.__all__)
__all__.extend(filters.__all__)
__all__.extend(iterators.__all__)
__all__.extend(processors.__all__)
__all__.extend(utilities.__all__)
