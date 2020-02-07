// Copyright (C) 2018 Chris N. Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CoordinateDofs.h"
#include <dolfinx/graph/AdjacencyList.h>

using namespace dolfinx;

//-----------------------------------------------------------------------------
mesh::CoordinateDofs::CoordinateDofs(
    const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        point_dofs)
    : _coord_dofs(new graph::AdjacencyList<std::int32_t>(point_dofs))

{
  // Do nothing
}
//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t>& mesh::CoordinateDofs::entity_points()
{
  assert(_coord_dofs);
  return *_coord_dofs;
}
//-----------------------------------------------------------------------------
const graph::AdjacencyList<std::int32_t>&
mesh::CoordinateDofs::entity_points() const
{
  assert(_coord_dofs);
  return *_coord_dofs;
}
//-----------------------------------------------------------------------------
