# Copyright (C) 2020 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


def get_include_path():
    """Return path to nanobind wrapper header files"""
    import pathlib

    return pathlib.Path(__file__).parent
