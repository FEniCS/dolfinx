# -*- coding: utf-8 -*-
# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import cpp


class Expression(cpp.function.Expression):

    def __init__(self, eval_func, shape=()):
        # Without this, undefined behaviour might happen due pybind docs
        cpp.function.Expression.__init__(self, shape)
        self.shape = shape
        self.eval_func = eval_func

    def eval_cell(self, values, x, cell):
        self.eval_func(values, x, cell)
