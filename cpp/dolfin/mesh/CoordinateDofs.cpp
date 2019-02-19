// Copyright (C) 2018 Chris N Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CoordinateDofs.h"
#include "MeshConnectivity.h"

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
  _coord_dofs[dim] = std::make_shared<MeshConnectivity>(point_dofs.rows(),
                                                        point_dofs.cols());
  for (std::uint32_t i = 0; i < point_dofs.rows(); ++i)
    _coord_dofs[dim]->set(i, point_dofs.row(i));
  _cell_permutation = cell_permutation;
}
//-----------------------------------------------------------------------------
const MeshConnectivity& CoordinateDofs::entity_points(std::uint32_t dim) const
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
