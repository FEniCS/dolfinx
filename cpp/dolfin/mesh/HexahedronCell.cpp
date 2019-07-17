// Copyright (C) 2015 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "HexahedronCell.h"
#include "Cell.h"
#include "Facet.h"
#include "MeshEntity.h"
#include "Vertex.h"
#include <algorithm>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
double HexahedronCell::squared_distance(const mesh::Cell& cell,
                                        const Eigen::Vector3d& point) const
{

  throw std::runtime_error("Not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
Eigen::Vector3d HexahedronCell::normal(const mesh::Cell& cell,
                                       std::size_t facet) const
{
  throw std::runtime_error("Not implemented");
  return Eigen::Vector3d();
}
//-----------------------------------------------------------------------------
Eigen::Vector3d HexahedronCell::cell_normal(const mesh::Cell& cell) const
{
  throw std::runtime_error("Not implemented");
  return Eigen::Vector3d();
}
//-----------------------------------------------------------------------------
