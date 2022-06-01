# Copyright (C) 2022 Nathan Sime
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
import importlib
import pkgutil


def collect_subpackages_recursive(name):
    module = importlib.import_module(name)
    submodules = list(a.name for a in pkgutil.iter_modules(
        module.__path__, prefix=f"{name}."))
    subpackages = list(a.name for a in pkgutil.iter_modules(
        module.__path__, prefix=f"{name}.") if a.ispkg)
    for subpackage in subpackages:
        pkg_submodules = collect_subpackages_recursive(subpackage)
        submodules.extend(pkg_submodules)
    return list(set(submodules))


@pytest.mark.parametrize("module_name",
                         collect_subpackages_recursive("dolfinx"))
def test_all_implemented(module_name):
    module = importlib.import_module(module_name)
    if hasattr(module, "__all__"):
        for member in module.__all__:
            assert hasattr(module, member)
