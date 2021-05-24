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
    # remove any final suffix like '.dev0'
    version = '.'.join(version.split('.')[:3])
    cpp_version = dolfinx.__version__
    # remove any final suffix like '-rc0'
    cpp_version = cpp_version.split("-")[0]
    if version != cpp_version:
        raise RuntimeError("Incorrect installed version")
