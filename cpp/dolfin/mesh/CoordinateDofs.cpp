// Copyright (C) 2018 Chris N Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CoordinateDofs.h"
#include "Connectivity.h"

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
CoordinateDofs::CoordinateDofs(std::uint32_t tdim) : _coord_dofs(tdim + 1)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void CoordinateDofs::init(std::size_t dim,
                          const Eigen::Ref<const EigenRowArrayXXi32> point_dofs,
                          const std::vector<std::uint8_t>& cell_permutation)
{
  assert(dim < _coord_dofs.size());
  _coord_dofs[dim] = std::make_shared<Connectivity>(point_dofs);
  _cell_permutation = cell_permutation;
}
//-----------------------------------------------------------------------------
const Connectivity& CoordinateDofs::entity_points(std::uint32_t dim) const
{
  assert(dim < _coord_dofs.size());
  assert(_coord_dofs[dim]);
  return *_coord_dofs[dim];
}
//-----------------------------------------------------------------------------
const std::vector<std::uint8_t>& CoordinateDofs::cell_permutation() const
{
  return _cell_permutation;
}
//-----------------------------------------------------------------------------
