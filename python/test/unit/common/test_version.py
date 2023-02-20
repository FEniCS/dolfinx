# Copyright (C) 2021 Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pkg_resources

import dolfinx


def test_version():
    """Test that installed Python version matches c++ version"""
    version = pkg_resources.get_distribution("fenics-dolfinx").version
    # Change any final '.dev0' to '.0'
    version = version.replace('dev', '')
    cpp_version = dolfinx.__version__
    if version != cpp_version:
        raise RuntimeError(f"Incorrect installed version {version}, {cpp_version}")
