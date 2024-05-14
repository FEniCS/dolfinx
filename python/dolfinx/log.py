# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Logging module."""

# Import nanobind wrapped code intp dolfinx.log
from dolfinx.cpp.log import LogLevel, log, set_output_file  # noqa


def set_log_level(level):
    print(level)


def get_log_level():
    return "unknown"
