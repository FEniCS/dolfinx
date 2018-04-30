// Copyright (C) 2018 Chris N Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CoordinateDofs.h"

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
CoordinateDofs::CoordinateDofs(std::uint32_t tdim)
{
  // Create mappings from entities to points for all topological dimensions
  // but only fill in cell_dofs for now
  for (unsigned int i = 0; i <= tdim; ++i)
    _coord_dofs.push_back(MeshConnectivity(i, 0));
}
//-----------------------------------------------------------------------------
void CoordinateDofs::init(std::size_t dim,
                          Eigen::Ref<const EigenRowArrayXXi32> point_dofs,
                          const std::vector<std::uint8_t>& cell_permutation)
{
  assert(dim < _coord_dofs.size());

  _coord_dofs[dim].init(point_dofs.rows(), point_dofs.cols());
  for (std::uint32_t i = 0; i < point_dofs.rows(); ++i)
    _coord_dofs[dim].set(i, point_dofs.row(i).data());

  _cell_permutation = cell_permutation;
}
//-----------------------------------------------------------------------------
