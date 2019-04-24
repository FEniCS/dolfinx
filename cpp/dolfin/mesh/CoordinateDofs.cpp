// Copyright (C) 2018 Chris N Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CoordinateDofs.h"
#include "Connectivity.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
mesh::CoordinateDofs::CoordinateDofs(
    int tdim,
    const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>
        point_dofs,
    const std::vector<std::uint8_t>& cell_permutation)
    : _coord_dofs(tdim + 1), _cell_permutation(cell_permutation)

{
  _coord_dofs[tdim] = std::make_shared<Connectivity>(point_dofs);
}
//-----------------------------------------------------------------------------
const mesh::Connectivity&
mesh::CoordinateDofs::entity_points(std::uint32_t dim) const
{
  assert(dim < _coord_dofs.size());
  assert(_coord_dofs[dim]);
  return *_coord_dofs[dim];
}
//-----------------------------------------------------------------------------
const std::vector<std::uint8_t>& mesh::CoordinateDofs::cell_permutation() const
{
  return _cell_permutation;
}
//-----------------------------------------------------------------------------
