// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "dolfinx_wrappers/geometry.h"
#include <nanobind/nanobind.h>

namespace nb = nanobind;
namespace dolfinx_wrappers
{
void geometry(nb::module_& m)
{
  declare_bbtree<float>(m, "float32");
  declare_bbtree<double>(m, "float64");
}
} // namespace dolfinx_wrappers
