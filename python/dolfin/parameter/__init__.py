# -*- coding: utf-8 -*-
# Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""FIXME: document"""

import dolfin.cpp as cpp
from ffc import default_jit_parameters
from dolfin.cpp.parameter import parameters, Parameters


#  Extend cpp.Parameters with a __getitem__ method
def __getitem__(self, key):
    if self.has_parameter(key):
        p = self._get_parameter(key)
        return p.value()
    elif self.has_parameter_set(key):
        # FIXME: I think we want to return the parameter set rather than a copy?
        p = self._get_parameter_set(key)
        return p
        # np = cpp.parameter.Parameters(p)
        # return np
    else:
        raise RuntimeError("Invalid parameter: {}".format(key))


# FIXME: This is probably better handled on the C++ side using
# py::dict
def update(self, params):
    if isinstance(params, cpp.parameter.Parameters):
        self._update(params)
    elif isinstance(params, dict):
        for key in params:
            if isinstance(params[key], dict):
                self[key].update(params[key])
            else:
                self[key] = params[key]
    else:
        raise ValueError("Parameters or dict")


# Extend the cpp.parameter.Parameters class and clean-up
cpp.parameter.Parameters.__getitem__ = __getitem__
cpp.parameter.Parameters.update = update
del __getitem__, update


def ffc_default_parameters():
    """Get default parameters of FFC"""
    # Get dict with defaults

    # FIXME: intialising MPI because setting parameters makes MPI
    # calls, possibly via the log systems. Needs to be fixed.
    cpp.MPI.init()

    d = default_jit_parameters()
    p = Parameters("form_compiler")

    typemap = {"quadrature_rule": "", "quadrature_degree": 0,
               "precision": 0}

    # Add the rest
    for key, value in d.items():
        if value is None:
            p.add(key, typemap[key])
            p[key] = None
        else:
            p.add(key, value)

    return p


# Add form compiler parameters to global parameter set
if not parameters.has_parameter_set("form_compiler"):
    parameters.add(ffc_default_parameters())
