// Copyright (C) 2007-2008 Kristian B. Oelgaard
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PointCell.h"
#include "Cell.h"
#include "Facet.h"
#include "MeshEntity.h"
#include "Vertex.h"

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
int PointCell::num_entities(int dim) const
{
  switch (dim)
  {
  case 0:
    return 1; // vertices
  default:
    throw std::runtime_error("Illegal dimension");
  }

  return 0;
}
//-----------------------------------------------------------------------------
double PointCell::squared_distance(const Cell& cell,
                                   const Eigen::Vector3d& point) const
{
  throw std::runtime_error("Not defined");
  return 0.0;
}
//-----------------------------------------------------------------------------
Eigen::Vector3d PointCell::normal(const Cell& cell, std::size_t facet) const
{
  throw std::runtime_error("Not defined");
  return Eigen::Vector3d();
}
//-----------------------------------------------------------------------------
Eigen::Vector3d PointCell::cell_normal(const Cell& cell) const
{
  throw std::runtime_error("Not defined");
  return Eigen::Vector3d();
}
//-----------------------------------------------------------------------------
std::size_t PointCell::find_edge(std::size_t i, const Cell& cell) const
{
  throw std::runtime_error("Not defined");
  return 0;
}
//-----------------------------------------------------------------------------
