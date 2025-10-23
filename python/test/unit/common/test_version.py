# Copyright (C) 2021-2025 Chris Richardson, Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from packaging.version import parse

import dolfinx


def test_version():
    """Test that installed Python version matches C++ version.

    C++ DOLFINx follows `major.minor.micro` with a development version
    denoted `major.minor.micro.0`.

    Python DOLFINx follows `major.minor.micro` with the append of `.devx`
    and `.postx` denoting development and post-release.
    """
    cpp_version = parse(dolfinx.cpp.__version__)
    python_version = parse(dolfinx.__version__)

    assert cpp_version.major == python_version.major
    assert cpp_version.minor == python_version.minor
    assert cpp_version.micro == python_version.micro

    cpp_is_dev = True if len(cpp_version.release) == 4 else False
    assert cpp_is_dev == python_version.is_devrelease
