# -*- coding: utf-8 -*-
# Copyright (C) 2019 Michal Habera and Chris Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from typing import Sequence, Union

import numpy

import dolfin.cpp
import ufl


class Constant(ufl.Constant):
    def __init__(self, domain, value: Union[numpy.ndarray, Sequence, float]):
        """A constant with respect to a domain.

        Parameters
        ----------
        domain : DOLFIN or UFL mesh
        value
            Value of the constant.
        """
        np_value = numpy.asarray(value)
        super().__init__(domain, np_value.shape)
        self._cpp_object = dolfin.cpp.function.Constant(np_value.shape, np_value.flatten())

    @property
    def value(self):
        """Returns value of the constant."""
        return self._cpp_object.value()

    @value.setter
    def value(self, v):
        numpy.copyto(self._cpp_object.value(), numpy.asarray(v))
