# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Logging module."""

# Import nanobind wrapped code into dolfinx.log
from dolfinx.cpp.log import (
    LogLevel,  # noqa
    get_log_level,  # noqa
    log,  # noqa
    set_log_level,  # noqa
    set_output_file,  # noqa
)
