# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Logging module"""

# Import pybind11 wrapped code intp dolfinx.log
from dolfinx.cpp.log import (LogLevel, get_log_level, log,  # noqa
                             set_log_level, set_output_file)
