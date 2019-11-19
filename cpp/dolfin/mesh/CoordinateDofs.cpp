// Copyright (C) 2018 Chris N. Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CoordinateDofs.h"
#include "Connectivity.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
mesh::CoordinateDofs::CoordinateDofs(
    const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        point_dofs)
    : _coord_dofs(new Connectivity(point_dofs))

{
  // Do nothing
}
//-----------------------------------------------------------------------------
mesh::Connectivity& mesh::CoordinateDofs::entity_points()
{
  assert(_coord_dofs);
  return *_coord_dofs;
}
//-----------------------------------------------------------------------------
const mesh::Connectivity& mesh::CoordinateDofs::entity_points() const
{
  assert(_coord_dofs);
  return *_coord_dofs;
}
//-----------------------------------------------------------------------------
