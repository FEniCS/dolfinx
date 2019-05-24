// Copyright (C) 2006-2015 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Facet.h"
#include "Cell.h"
#include "IntervalCell.h"
#include "TriangleCell.h"

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
Eigen::Vector3d Facet::normal() const
{
  const std::size_t D = _mesh->topology().dim();
  _mesh->create_entities(D - 1);
  _mesh->create_connectivity(D - 1, D);

  // Get cell to which face belong (first cell when there is more than one)
  const Cell cell(*_mesh, this->entities(D)[0]);

  // Get local index of facet with respect to the cell
  const std::size_t local_facet = cell.index(*this);

  return cell.normal(local_facet);
}
//-----------------------------------------------------------------------------
double Facet::squared_distance(const Eigen::Vector3d& point) const
{
  if (_dim == 1)
  {
    // Extract vertices
    const Geometry& geometry = _mesh->geometry();
    const std::int32_t* vertices = entities(0);
    const Eigen::Vector3d a = geometry.x(vertices[0]);
    const Eigen::Vector3d b = geometry.x(vertices[1]);

    // Compute squared distance
    return IntervalCell::squared_distance(point, a, b);
  }
  else if (_dim == 2)
  {
    // Extract vertices
    const Geometry& geometry = _mesh->geometry();
    const std::int32_t* vertices = entities(0);
    const Eigen::Vector3d a = geometry.x(vertices[0]);
    const Eigen::Vector3d b = geometry.x(vertices[1]);
    const Eigen::Vector3d c = geometry.x(vertices[2]);

    // Compute squared distance
    return TriangleCell::squared_distance(point, a, b, c);
  }

  throw std::runtime_error(" Compute (squared) distance to facet not "
                           "implemented for facets of dimension "
                           + std::to_string(_dim));

  return 0.0;
}
//-----------------------------------------------------------------------------
bool Facet::exterior() const
{
  const std::size_t D = _mesh->topology().dim();
  if (this->num_global_entities(D) == 1)
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
