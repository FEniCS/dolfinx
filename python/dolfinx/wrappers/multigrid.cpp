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

#include "dolfinx_wrappers/array.h"

namespace nb = nanobind;

namespace dolfinx_wrappers
{

void multigrid(nb::module_& m)
{
  m.def(
      "inclusion_mapping",
      [](const dolfinx::mesh::Mesh<double>& mesh_from,
         const dolfinx::mesh::Mesh<double>& mesh_to, bool allow_all_to_all)
      {
        std::vector<std::int64_t> map
            = dolfinx::multigrid::inclusion_mapping<double>(mesh_from, mesh_to,
                                                            allow_all_to_all);
        return dolfinx_wrappers::as_nbarray(std::move(map));
      },
      nb::arg("mesh_from"), nb::arg("mesh_to"),
      "Computes inclusion mapping between two meshes");

  m.def(
      "create_sparsity_pattern",
      [](const dolfinx::fem::FunctionSpace<double>& V_from,
         const dolfinx::fem::FunctionSpace<double>& V_to,
         nb::ndarray<const std::int64_t, nb::numpy>& inclusion_map)
      {
        // TODO: cahnge to accepting range;
        auto vec = std::vector(inclusion_map.data(),
                               inclusion_map.data() + inclusion_map.size());
        return dolfinx::multigrid::create_sparsity_pattern<double>(V_from, V_to,
                                                                   vec);
      },
      nb::arg("V_from"), nb::arg("V_to"), nb::arg("inclusion_map"));

  m.def(
      "assemble_transfer_matrix",
      [](dolfinx::la::MatrixCSR<double>& A,
         const dolfinx::fem::FunctionSpace<double>& V_from,
         const dolfinx::fem::FunctionSpace<double>& V_to,
         const std::vector<std::int64_t>& inclusion_map,
         std::function<double(std::int32_t)> weight)
      {
        dolfinx::multigrid::assemble_transfer_matrix(
            A.mat_set_values(), V_from, V_to, inclusion_map, weight);
      },
      nb::arg("A"), nb::arg("V_from"), nb::arg("V_to"),
      nb::arg("inclusion_map"), nb::arg("weight"),
      "Assembles a transfer matrix between two function spaces.");
}

} // namespace dolfinx_wrappers
