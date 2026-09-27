// Copyright (C) 2013-2026 Johan Hake, Jan Blechta, Garth N. Wells and Paul T.
// Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "DofMap.h"
#include "Form.h"
#include "FunctionSpace.h"
#include "sparsitybuild.h"
#include <array>
#include <concepts>
#include <cstddef>
#include <dolfinx/common/Timer.h>
#include <dolfinx/la/SparsityPattern.h>
#include <functional>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

/// @file sparsitypattern.h
/// @brief Functions for constructing finite element sparsity patterns.

namespace dolfinx::fem
{
/// @brief Extract test (0) and trial (1) function spaces pairs for each
/// bilinear form for a rectangular array of forms.
///
/// @param[in] a A rectangular block on bilinear forms.
/// @return Rectangular array of the same shape as `a` with a pair of
/// function spaces in each array entry. If a form is null, then the
/// returned function space pair is (null, null).
template <dolfinx::scalar T, std::floating_point U>
std::vector<std::vector<std::array<std::shared_ptr<const FunctionSpace<U>>, 2>>>
extract_function_spaces(const std::vector<std::vector<const Form<T, U>*>>& a)
{
  std::vector<
      std::vector<std::array<std::shared_ptr<const FunctionSpace<U>>, 2>>>
      spaces(
          a.size(),
          std::vector<std::array<std::shared_ptr<const FunctionSpace<U>>, 2>>(
              a.front().size()));
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    for (std::size_t j = 0; j < a[i].size(); ++j)
    {
      if (const Form<T, U>* form = a[i][j]; form)
        spaces[i][j] = {form->function_spaces()[0], form->function_spaces()[1]};
    }
  }
  return spaces;
}

/// @brief Create a sparsity pattern for a given form.
/// @note The pattern is not finalised, i.e. the caller is responsible
/// for calling SparsityPattern::assemble.
/// @param[in] a A bilinear form
/// @return The corresponding sparsity pattern
template <dolfinx::scalar T, std::floating_point U>
la::SparsityPattern create_sparsity_pattern(const Form<T, U>& a)
{
  std::shared_ptr mesh = a.mesh();
  assert(mesh);

  // Get index maps and block sizes from the DOF maps. Note that in
  // mixed-topology meshes, despite there being multiple DOF maps, the
  // index maps and block sizes are the same.
  std::array<std::reference_wrapper<const DofMap>, 2> dofmaps{
      *a.function_spaces().at(0)->dofmaps().front(),
      *a.function_spaces().at(1)->dofmaps().front()};

  const std::array index_maps{dofmaps[0].get().index_map,
                              dofmaps[1].get().index_map};
  const std::array bs
      = {dofmaps[0].get().index_map_bs(), dofmaps[1].get().index_map_bs()};

  la::SparsityPattern pattern(mesh->comm(), index_maps, bs);
  build_sparsity_pattern(pattern, a);
  return pattern;
}

/// @brief Build a sparsity pattern for a given form.
/// @note The pattern is not finalised, i.e. the caller is responsible
/// for calling SparsityPattern::assemble.
/// @param[in] pattern The sparsity pattern to add to
/// @param[in] a A bilinear form
template <dolfinx::scalar T, std::floating_point U>
void build_sparsity_pattern(la::SparsityPattern& pattern, const Form<T, U>& a)
{
  if (a.rank() != 2)
  {
    throw std::invalid_argument(
        "Cannot create sparsity pattern. Form is not a bilinear.");
  }

  std::shared_ptr mesh = a.mesh();
  assert(mesh);
  std::shared_ptr mesh0 = a.function_spaces().at(0)->mesh();
  assert(mesh0);
  std::shared_ptr mesh1 = a.function_spaces().at(1)->mesh();
  assert(mesh1);

  const std::set<IntegralType> types = a.integral_types();
  if (types.find(IntegralType::interior_facet) != types.end()
      or types.find(IntegralType::exterior_facet) != types.end())
  {
    // FIXME: cleanup these calls? Some of the happen internally again.
    int tdim = mesh->topology()->dim();
    mesh->topology_mutable()->create_entities(tdim - 1);
    mesh->topology_mutable()->create_connectivity(tdim - 1, tdim);
  }

  common::Timer t0("Build sparsity");

  auto extract_cells = [](std::span<const std::int32_t> facets)
  {
    assert(facets.size() % 2 == 0);
    std::vector<std::int32_t> cells;
    cells.reserve(facets.size() / 2);
    for (std::size_t i = 0; i < facets.size(); i += 2)
      cells.push_back(facets[i]);
    return cells;
  };

  const int num_cell_types = mesh->topology()->cell_types().size();
  for (int cell_type_idx = 0; cell_type_idx < num_cell_types; ++cell_type_idx)
  {
    std::array<std::reference_wrapper<const DofMap>, 2> dofmaps{
        *a.function_spaces().at(0)->dofmaps().at(cell_type_idx),
        *a.function_spaces().at(1)->dofmaps().at(cell_type_idx)};

    // Create and build sparsity pattern
    for (auto type : types)
    {
      switch (type)
      {
      case IntegralType::cell:
        for (int i = 0; i < a.num_integrals(type, cell_type_idx); ++i)
        {
          sparsitybuild::cells(
              pattern,
              std::pair{a.domain_arg(type, 0, i, cell_type_idx),
                        a.domain_arg(type, 1, i, cell_type_idx)},
              {{dofmaps[0], dofmaps[1]}});
        }
        break;
      case IntegralType::interior_facet:
        for (int i = 0; i < a.num_integrals(type, cell_type_idx); ++i)
        {
          sparsitybuild::interior_facets(
              pattern,
              {extract_cells(a.domain_arg(type, 0, i, 0)),
               extract_cells(a.domain_arg(type, 1, i, 0))},
              {{dofmaps[0], dofmaps[1]}});
        }
        break;
      case IntegralType::exterior_facet:
      case IntegralType::ridge:
      case IntegralType::vertex:
        for (int i = 0; i < a.num_integrals(type, cell_type_idx); ++i)
        {
          sparsitybuild::cells(
              pattern,
              std::pair{extract_cells(a.domain_arg(type, 0, i, 0)),
                        extract_cells(a.domain_arg(type, 1, i, 0))},
              {{dofmaps[0], dofmaps[1]}});
        }
        break;
      default:
        throw std::invalid_argument("Unsupported integral type");
      }
    }
  }

  t0.stop();
}
} // namespace dolfinx::fem
