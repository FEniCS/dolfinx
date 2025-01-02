// Copyright (C) 2024 Paul T. Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cstdint>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/multigrid/inclusion.h>

#include "array.h"

namespace nb = nanobind;

namespace dolfinx_wrappers
{

void multigrid(nb::module_& m)
{
  m.def(
      "inclusion_mapping",
      [](const dolfinx::mesh::Mesh<double>& mesh_from,
         const dolfinx::mesh::Mesh<double>& mesh_to)
      {
        std::vector<std::int64_t> map
            = dolfinx::multigrid::inclusion_mapping<double>(mesh_from, mesh_to);
        return dolfinx_wrappers::as_nbarray(std::move(map));
      },
      nb::arg("mesh_from"), nb::arg("mesh_to"),
      "Computes inclusion mapping between two meshes");
}

} // namespace dolfinx_wrappers
