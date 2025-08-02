// Copyright (C) 2025 Paul T. Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <concepts>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/multigrid/inclusion.h>

#include "dolfinx_wrappers/array.h"

namespace nb = nanobind;

namespace dolfinx_wrappers
{

template <std::floating_point T>
void declare_inlcusion_mapping(nb::module_& m, const std::string& type)
{
  
  m.def(
      ("inclusion_mapping_" + type).c_str(),
      [](const dolfinx::mesh::Mesh<T>& mesh_from,
         const dolfinx::mesh::Mesh<T>& mesh_to)
      {
        return dolfinx_wrappers::as_nbarray(
            dolfinx::multigrid::inclusion_mapping<T>(mesh_from, mesh_to));
      },
      nb::arg("mesh_from"), nb::arg("mesh_to"),
      "Computes inclusion mapping between two meshes");
}

void multigrid(nb::module_& m)
{
  declare_inlcusion_mapping<float>(m, "float32");
  declare_inlcusion_mapping<double>(m, "float64");
}

} // namespace dolfinx_wrappers
