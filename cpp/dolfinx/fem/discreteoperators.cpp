// Copyright (C) 2015-2022 Garth N. Wells, Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "discreteoperators.h"
#include "sparsitybuild.h"

using namespace dolfinx;

//-----------------------------------------------------------------------------
la::SparsityPattern
fem::create_sparsity_discrete_gradient(const fem::FunctionSpace& V0,
                                       const fem::FunctionSpace& V1)
{
  // Get mesh
  std::shared_ptr<const mesh::Mesh> mesh = V0.mesh();
  assert(mesh);

  // Check that mesh is the same for both function spaces
  assert(V1.mesh());
  if (mesh != V1.mesh())
  {
    throw std::runtime_error("Compute discrete gradient operator. Function "
                             "spaces do not share the same mesh");
  }

  // Check that output space uses covariant Piola while input space uses
  // identity
  auto e0 = V0.element();
  auto e1 = V1.element();
  assert(e0->map_type() == basix::maps::type::covariantPiola);
  assert(e1->map_type() == basix::maps::type::identity);

  // Copy index maps from dofmaps
  std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps
      = {{V0.dofmap()->index_map, V1.dofmap()->index_map}};
  std::array<int, 2> block_sizes
      = {V0.dofmap()->index_map_bs(), V1.dofmap()->index_map_bs()};
  std::vector<std::array<std::int64_t, 2>> local_range
      = {index_maps[0]->local_range(), index_maps[1]->local_range()};
  assert(block_sizes[0] == block_sizes[1]);

  // Create and assemble sparsity pattern
  la::SparsityPattern pattern(mesh->comm(), index_maps, block_sizes);
  std::array<const std::reference_wrapper<const fem::DofMap>, 2> dofmaps
      = {*V0.dofmap(), *V1.dofmap()};
  sparsitybuild::cells(pattern, mesh->topology(), dofmaps);
  pattern.assemble();
  return pattern;
};
