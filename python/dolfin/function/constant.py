# -*- coding: utf-8 -*-
# Copyright (C) 2019 Michal Habera, Chris Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy

import ufl
import dolfin.cpp


class Constant(ufl.Constant):
    def __init__(self, domain, value):
        np_value = numpy.asarray(value)
        super().__init__(domain, np_value.shape)
        self._cpp_object = dolfin.cpp.function.Constant(np_value.flatten(), np_value.shape)

    @property
    def value(self):
        val = self._cpp_object.array()
        if (len(val) == 1):
            return val[0]
        else:
            return val

    @value.setter
    def value(self, val):
        np_value = numpy.asarray(val)
        if (np_value.shape != self.ufl_shape):
            raise RuntimeError("Shape of the value must agree with shape of the Constant ({}).".format(self.ufl_shape))
        self._cpp_object.value = np_value.flatten()
