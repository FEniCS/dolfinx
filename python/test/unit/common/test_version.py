# Copyright (C) 2021-2022 Chris Richardson, Matthew Scroggs
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pkg_resources
import os
import re
import pytest
import dolfinx


def test_version():
    """Test that installed Python version matches c++ version"""
    version = pkg_resources.get_distribution("fenics-dolfinx").version
    # Change any final '.dev0' to '.0'
    version = version.replace('dev', '')
    cpp_version = dolfinx.__version__
    if version != cpp_version:
        raise RuntimeError("Incorrect installed version")


def test_version_numbering():
    py_version = pkg_resources.get_distribution("fenics-dolfinx").version
    cpp_version = py_version.replace('dev', '')

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../..")
    if not os.path.isfile(os.path.join(path, "COPYING")):
        pytest.skip("This test can only be run from the source directory.")

    for file in ["cpp/CMakeLists.txt"]:
        print(f"Checking version numbers in {file}.")

        with open(os.path.join(path, file)) as f:
            content = f.read()
        matches = re.findall(r"(?:(?:VERSION)|(?:version))[\s=]+([\"'])(.+?)\1", content)
        assert len(matches) > 0
        for m in matches:
            assert m[1] == cpp_version

    for file in ["python/setup.py"]:
        print(f"Checking version numbers in {file}.")

        with open(os.path.join(path, file)) as f:
            content = f.read()
        matches = re.findall(r"(?:(?:VERSION)|(?:version))[\s=]+([\"'])(.+?)\1", content)
        assert len(matches) > 0
        for m in matches:
            assert m[1] == py_version
