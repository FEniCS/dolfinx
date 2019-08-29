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
        return numpy.array(self._cpp_object, copy=False)

    @value.setter
    def value(self, val):
        value_ref = numpy.array(self._cpp_object, copy=False)
        numpy.copyto(value_ref, val)
