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
#include <dolfin/log/log.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
std::size_t HexahedronCell::dim() const { return 3; }
//-----------------------------------------------------------------------------
std::size_t HexahedronCell::num_entities(std::size_t dim) const
{
  switch (dim)
  {
  case 0:
    return 8; // vertices
  case 1:
    return 12; // edges
  case 2:
    return 6; // faces
  case 3:
    return 1; // cells
  default:
    log::dolfin_error("HexahedronCell.cpp",
                      "access number of entities of hexahedron cell",
                      "Illegal topological dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
std::size_t HexahedronCell::num_vertices(std::size_t dim) const
{
  switch (dim)
  {
  case 0:
    return 1; // vertices
  case 1:
    return 2; // edges
  case 2:
    return 4; // faces
  case 3:
    return 8; // cells
  default:
    log::dolfin_error(
        "HexahedronCell.cpp",
        "access number of vertices for subsimplex of hexahedron cell",
        "Illegal topological dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
void HexahedronCell::create_entities(boost::multi_array<std::int32_t, 2>& e,
                                     std::size_t dim,
                                     const std::int32_t* v) const
{
  // We need to know how to create edges and faces
  switch (dim)
  {
  case 1:
    // Resize data structure
    e.resize(boost::extents[12][2]);

    // Create the 12 edges
    e[0][0] = v[0];
    e[0][1] = v[1];
    e[1][0] = v[2];
    e[1][1] = v[3];
    e[2][0] = v[4];
    e[2][1] = v[5];
    e[3][0] = v[6];
    e[3][1] = v[7];
    e[4][0] = v[0];
    e[4][1] = v[2];
    e[5][0] = v[1];
    e[5][1] = v[3];
    e[6][0] = v[4];
    e[6][1] = v[6];
    e[7][0] = v[5];
    e[7][1] = v[7];
    e[8][0] = v[0];
    e[8][1] = v[4];
    e[9][0] = v[1];
    e[9][1] = v[5];
    e[10][0] = v[2];
    e[10][1] = v[6];
    e[11][0] = v[3];
    e[11][1] = v[7];
    break;
  case 2:
    // Resize data structure
    e.resize(boost::extents[6][4]);

    // Create the 6 faces
    e[0][0] = v[0];
    e[0][1] = v[1];
    e[0][2] = v[2];
    e[0][3] = v[3];
    e[1][0] = v[4];
    e[1][1] = v[5];
    e[1][2] = v[6];
    e[1][3] = v[7];
    e[2][0] = v[0];
    e[2][1] = v[1];
    e[2][2] = v[4];
    e[2][3] = v[5];
    e[3][0] = v[2];
    e[3][1] = v[3];
    e[3][2] = v[6];
    e[3][3] = v[7];
    e[4][0] = v[0];
    e[4][1] = v[2];
    e[4][2] = v[4];
    e[4][3] = v[6];
    e[5][0] = v[1];
    e[5][1] = v[3];
    e[5][2] = v[5];
    e[5][3] = v[7];
    break;
  default:
    log::dolfin_error(
        "HexahedronCell.cpp", "create entities of tetrahedron cell",
        "Don't know how to create entities of topological dimension %d", dim);
  }
}
//-----------------------------------------------------------------------------
double HexahedronCell::volume(const MeshEntity& cell) const
{
  if (cell.dim() != 2)
  {
    log::dolfin_error("HexahedronCell.cpp", "compute volume (area) of cell",
                      "Illegal mesh entity");
  }

  log::dolfin_error("HexahedronCell.cpp", "compute volume of hexahedron",
                    "Not implemented");

  return 0.0;
}
//-----------------------------------------------------------------------------
double HexahedronCell::circumradius(const MeshEntity& cell) const
{
  // Check that we get a cell
  if (cell.dim() != 2)
  {
    log::dolfin_error("HexahedronCell.cpp",
                      "compute circumradius of hexahedron cell",
                      "Illegal mesh entity");
  }

  log::dolfin_error("HexahedronCell.cpp",
                    "compute circumradius of hexahedron cell",
                    "Don't know how to compute diameter");

  dolfin_not_implemented();
  return 0.0;
}
//-----------------------------------------------------------------------------
double HexahedronCell::squared_distance(const mesh::Cell& cell,
                                        const geometry::Point& point) const
{
  dolfin_not_implemented();
  return 0.0;
}
//-----------------------------------------------------------------------------
double HexahedronCell::normal(const mesh::Cell& cell, std::size_t facet,
                              std::size_t i) const
{
  return normal(cell, facet)[i];
}
//-----------------------------------------------------------------------------
geometry::Point HexahedronCell::normal(const mesh::Cell& cell,
                                       std::size_t facet) const
{
  dolfin_not_implemented();
  return geometry::Point();
}
//-----------------------------------------------------------------------------
geometry::Point HexahedronCell::cell_normal(const mesh::Cell& cell) const
{
  dolfin_not_implemented();
  return geometry::Point();
}
//-----------------------------------------------------------------------------
double HexahedronCell::facet_area(const mesh::Cell& cell,
                                  std::size_t facet) const
{
  dolfin_not_implemented();

  return 0.0;
}
//-----------------------------------------------------------------------------
std::string HexahedronCell::description(bool plural) const
{
  if (plural)
    return "hexahedra";
  return "hexahedron";
}
//-----------------------------------------------------------------------------
