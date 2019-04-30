# -*- coding: utf-8 -*-
# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Logging module"""

# Import pybind11 wrapped code intp dolfin.log
from dolfin.cpp.log import log, info, set_log_level, get_log_level, LogLevel # noqa
