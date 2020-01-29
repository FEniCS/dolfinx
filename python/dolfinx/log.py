# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Logging module"""

# Import pybind11 wrapped code intp dolfinx.log
from dolfinx.cpp.log import log, set_output_file, set_log_level, get_log_level, LogLevel # noqa
