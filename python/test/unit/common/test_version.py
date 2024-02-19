# Copyright (C) 2021 Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from importlib.metadata import version

import dolfinx


def test_version():
    """Test that installed Python version matches C++ version."""
    py_version = version("fenics-dolfinx")
    # Change any final '.dev0' to '.0'
    py_version = py_version.replace("dev", "")
    cpp_version = dolfinx.__version__
    if py_version != cpp_version:
        raise RuntimeError(
            f"Incorrect versions. Python version: {py_version}, Core version: {cpp_version}"
        )
