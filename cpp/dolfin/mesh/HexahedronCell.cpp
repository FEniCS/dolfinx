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
void HexahedronCell::create_entities(
    Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        e,
    std::size_t dim, const std::int32_t* v) const
{
  // We need to know how to create edges and faces
  switch (dim)
  {
  case 1:
    // Resize data structure
    e.resize(12, 2);

    // Create the 12 edges
    e(0, 0) = v[0];
    e(0, 1) = v[1];
    e(1, 0) = v[2];
    e(1, 1) = v[3];
    e(2, 0) = v[4];
    e(2, 1) = v[5];
    e(3, 0) = v[6];
    e(3, 1) = v[7];
    e(4, 0) = v[0];
    e(4, 1) = v[2];
    e(5, 0) = v[1];
    e(5, 1) = v[3];
    e(6, 0) = v[4];
    e(6, 1) = v[6];
    e(7, 0) = v[5];
    e(7, 1) = v[7];
    e(8, 0) = v[0];
    e(8, 1) = v[4];
    e(9, 0) = v[1];
    e(9, 1) = v[5];
    e(10, 0) = v[2];
    e(10, 1) = v[6];
    e(11, 0) = v[3];
    e(11, 1) = v[7];
    break;
  case 2:
    // Resize data structure
    e.resize(6, 4);

    // Create the 6 faces
    e(0, 0) = v[0];
    e(0, 1) = v[1];
    e(0, 2) = v[2];
    e(0, 3) = v[3];
    e(1, 0) = v[4];
    e(1, 1) = v[5];
    e(1, 2) = v[6];
    e(1, 3) = v[7];
    e(2, 0) = v[0];
    e(2, 1) = v[1];
    e(2, 2) = v[4];
    e(2, 3) = v[5];
    e(3, 0) = v[2];
    e(3, 1) = v[3];
    e(3, 2) = v[6];
    e(3, 3) = v[7];
    e(4, 0) = v[0];
    e(4, 1) = v[2];
    e(4, 2) = v[4];
    e(4, 3) = v[6];
    e(5, 0) = v[1];
    e(5, 1) = v[3];
    e(5, 2) = v[5];
    e(5, 3) = v[7];
    break;
  default:
    throw std::runtime_error("Illegal topological dimension. Must be 1 or 2.");
  }
}
//-----------------------------------------------------------------------------
double HexahedronCell::circumradius(const MeshEntity& cell) const
{
  // Check that we get a cell
  if (cell.dim() != 2)
  {
    throw std::runtime_error("Illegal topological dimension");
  }

  throw std::runtime_error("Not Implemented");

  return 0.0;
}
//-----------------------------------------------------------------------------
double HexahedronCell::squared_distance(const mesh::Cell& cell,
                                        const Eigen::Vector3d& point) const
{

  throw std::runtime_error("Not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
double HexahedronCell::normal(const mesh::Cell& cell, std::size_t facet,
                              std::size_t i) const
{
  return normal(cell, facet)[i];
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
double HexahedronCell::facet_area(const mesh::Cell& cell,
                                  std::size_t facet) const
{
  throw std::runtime_error("Not implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
