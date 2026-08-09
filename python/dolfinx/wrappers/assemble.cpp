// Copyright (C) 2017-2021 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "dolfinx_wrappers/assemble.h"
#include <nanobind/nanobind.h>

namespace nb = nanobind;
namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

namespace dolfinx_wrappers
{

void assemble(nb::module_& m)
{
  // dolfinx::fem::assemble
  declare_assembly_functions<float, float>(m);
  declare_assembly_functions<double, double>(m);
  declare_assembly_functions<std::complex<float>, float>(m);
  declare_assembly_functions<std::complex<double>, double>(m);

  // interpolation_matrix/discrete_curl/discrete_gradient produce a real
  // (never complex) operator matrix, independent of the scalar type of
  // the FEM problem the FunctionSpace belongs to, so only one
  // instantiation per geometry type is registered.
  declare_discrete_operators<float, float>(m);
  declare_discrete_operators<double, double>(m);
}
} // namespace dolfinx_wrappers
