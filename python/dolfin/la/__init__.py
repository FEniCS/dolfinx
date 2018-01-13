# -*- coding: utf-8 -*-
# Copyright (C) 2017 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
from numpy import intc, int64
import dolfin.cpp as cpp


def la_index_dtype():
    """Return the numpy dtype equivalent to the type of la_index"""
    return intc if cpp.common.sizeof_la_index() == 4 else int64


def as_backend_type(x):
    """Return Matrix and Vector backend instance. Not required for other
    types as pybind11 automatically downcasts objects to the derived
    type.

    """
    if isinstance(x, cpp.la.Vector) or isinstance(x, cpp.la.Matrix) or isinstance(x, cpp.la.LinearOperator):
        return x.instance()
    else:
        return x


# Extend PETScVector
def __gt__(self, value):
    "Returns a boolean array with > status for all elements"
    if np.isscalar(value):
        return self.get_local() > value
    else:
        return self.get_local() > value.get_local()


cpp.la.PETScVector.__gt__ = __gt__
del __gt__


def __ge__(self, value):
    "Returns a boolean array with >= status for all elements"
    if np.isscalar(value):
        return self.get_local() >= value
    else:
        return self.get_local() >= value.get_local()


cpp.la.PETScVector.__ge__ = __ge__
del __ge__


def __lt__(self, value):
    "Returns a boolean array with < status for all elements"
    if np.isscalar(value):
        return self.get_local() < value
    else:
        return self.get_local() < value.get_local()


cpp.la.PETScVector.__lt__ = __lt__
del __lt__


def __le__(self, value):
    "Returns a boolean array with <= status for all elements"
    if np.isscalar(value):
        return self.get_local() <= value
    else:
        return self.get_local() <= value.get_local()


cpp.la.PETScVector.__le__ = __le__
del __le__


def __eq__(self, value):
    if np.isscalar(value):
        return self.get_local() == value
    else:
        return self.get_local() == value.get_local()


cpp.la.PETScVector.__eq__ = __eq__
del __eq__


def __iter__(self):
    for i in range(self.local_size()):
        yield self[i]


cpp.la.PETScVector.__iter__ = __iter__
del __iter__
