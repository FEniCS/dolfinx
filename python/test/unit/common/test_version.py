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
    if version != dolfinx.__version__:
        raise RuntimeError("Incorrect installed version")
