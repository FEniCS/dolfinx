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
std::size_t PointCell::dim() const { return 0; }
//-----------------------------------------------------------------------------
std::size_t PointCell::num_entities(std::size_t dim) const
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
std::size_t PointCell::num_vertices(std::size_t dim) const
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
void PointCell::create_entities(
    Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        e,
    std::size_t dim, const std::int32_t* v) const
{
  throw std::runtime_error("Not defined");
}
//-----------------------------------------------------------------------------
double PointCell::volume(const MeshEntity& triangle) const
{
  throw std::runtime_error("Not defined");
  return 0.0;
}
//-----------------------------------------------------------------------------
double PointCell::circumradius(const MeshEntity& point) const
{
  throw std::runtime_error("Not defined");
  return 0.0;
}
//-----------------------------------------------------------------------------
double PointCell::squared_distance(const Cell& cell,
                                   const Eigen::Vector3d& point) const
{
  throw std::runtime_error("Not defined");
  return 0.0;
}
//-----------------------------------------------------------------------------
double PointCell::normal(const Cell& cell, std::size_t facet,
                         std::size_t i) const
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
double PointCell::facet_area(const Cell& cell, std::size_t facet) const
{
  throw std::runtime_error("Not defined");
  return 0.0;
}
//-----------------------------------------------------------------------------
std::string PointCell::description(bool plural) const
{
  if (plural)
    return "points";
  return "points";
}
//-----------------------------------------------------------------------------
std::size_t PointCell::find_edge(std::size_t i, const Cell& cell) const
{
  throw std::runtime_error("Not defined");
  return 0;
}
//-----------------------------------------------------------------------------
