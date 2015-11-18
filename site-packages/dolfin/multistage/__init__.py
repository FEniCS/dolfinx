# -*- coding: utf-8 -*-
"""The multistage module of dolfin"""
from dolfin.multistage import multistagescheme
from dolfin.multistage import multistagesolvers
from dolfin.multistage import rushlarsenschemes

from .multistagescheme import *
from .multistagesolvers import *
from .rushlarsenschemes import *

# NOTE: The automatic documentation system in DOLFIN requires to _not_ define
# NOTE: classes or functions within this file. Use separate modules for that
# NOTE: purpose.

__all__ = multistagescheme.__all__ + multistagesolvers.__all__ + \
          rushlarsenschemes.__all__
