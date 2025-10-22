# Copyright (C) 2021-2025 Chris Richardson, Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import dolfinx


def test_version():
    """Test that installed Python version matches C++ version."""
    # Change any final '.dev0' to '.0'
    py_version = dolfinx.__version__.replace("dev0", "0")
    cpp_version = dolfinx.cpp.__version__
    if py_version != cpp_version:
        raise RuntimeError(
            f"Incorrect versions. Python version: {py_version}, Core version: {cpp_version}"
        )
