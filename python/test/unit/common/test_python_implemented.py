# Copyright (C) 2022 Nathan Sime
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import importlib
import pkgutil


def collect_pkg_modules_recursive(name):
    module = importlib.import_module(name)
    submodules = list(a.name for a in pkgutil.iter_modules(
        module.__path__, prefix=f"{name}."))
    subpackages = list(a.name for a in pkgutil.iter_modules(
        module.__path__, prefix=f"{name}.") if a.ispkg)
    for subpackage in subpackages:
        pkg_submodules = collect_pkg_modules_recursive(subpackage)
        submodules.extend(pkg_submodules)
    return list(set(submodules))


def test_all_implemented():
    module_names = collect_pkg_modules_recursive("dolfinx")
    for module_name in module_names:
        module = importlib.import_module(module_name)
        if hasattr(module, "__all__"):
            for member in module.__all__:
                assert hasattr(module, member)
