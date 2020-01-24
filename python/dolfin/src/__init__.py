# Copyright (C) 2020 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

def get_path():
    """Return path to pybind11 wrapper header files"""
    import pathlib
    return pathlib.Path(__file__).parent

