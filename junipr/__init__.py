"""The JUNIPR package."""
from __future__ import absolute_import

import tensorflow as tf

# import toplevel submodules
from . import tfrecord
from . import utils
from . import config
from . import junipr

# import toplevel attributes
from .config import *
from .utils import *
from .tfrecord import *
from .junipr import *

__all__ = (config.__all__ + 
           utils.__all__+
           tfrecord.__all__ +
           junipr.__all__)

__version__ = '0.1'