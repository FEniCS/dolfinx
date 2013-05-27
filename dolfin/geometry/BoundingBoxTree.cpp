// Copyright (C) 2013 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2013-04-09
// Last changed: 2013-05-27

#include <dolfin/log/log.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/mesh/Mesh.h>
#include "BoundingBoxTree1D.h"
#include "BoundingBoxTree2D.h"
#include "BoundingBoxTree3D.h"
#include "BoundingBoxTree.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundingBoxTree::~BoundingBoxTree()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::build(const Mesh& mesh)
{
  build(mesh, mesh.topology().dim());
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::build(const Mesh& mesh, std::size_t tdim)
{
  // Select implementation
  switch (mesh.geometry().dim())
  {
  case 1:
    _tree.reset(new BoundingBoxTree1D());
    break;
  case 2:
    _tree.reset(new BoundingBoxTree2D());
    break;
  case 3:
    _tree.reset(new BoundingBoxTree3D());
    break;
  default:
    dolfin_error("BoundingBoxTree.cpp",
                 "build bounding box tree",
                 "Not implemented for geometric dimension %d",
                 mesh.geometry().dim());
  }

  // Build tree
  dolfin_assert(_tree);
  _tree->build(mesh, tdim);
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::build(const std::vector<Point>& points, std::size_t gdim)
{
  // Select implementation
  switch (gdim)
  {
  case 1:
    _tree.reset(new BoundingBoxTree1D());
    break;
  case 2:
    _tree.reset(new BoundingBoxTree2D());
    break;
  case 3:
    _tree.reset(new BoundingBoxTree3D());
    break;
  default:
    dolfin_error("BoundingBoxTree.cpp",
                 "build bounding box tree",
                 "Not implemented for geometric dimension %d",
                 gdim);
  }

  // Build tree
  dolfin_assert(_tree);
  _tree->build(points);
}
//-----------------------------------------------------------------------------
std::vector<mesh_index>
BoundingBoxTree::compute_collisions(const Point& point) const
{
  // Check that tree has been built
  check_built();

  // Delegate call to implementation
  dolfin_assert(_tree);
  return _tree->compute_collisions(point);
}
//-----------------------------------------------------------------------------
std::vector<mesh_index>
BoundingBoxTree::compute_entity_collisions(const Point& point,
                                           const Mesh& mesh) const
{
  // Check that tree has been built
  check_built();

  // Delegate call to implementation
  dolfin_assert(_tree);
  return _tree->compute_entity_collisions(point, mesh);
}
//-----------------------------------------------------------------------------
mesh_index
BoundingBoxTree::compute_first_collision(const Point& point) const
{
  // Check that tree has been built
  check_built();

  // Delegate call to implementation
  dolfin_assert(_tree);
  return _tree->compute_first_collision(point);
}
//-----------------------------------------------------------------------------
mesh_index
BoundingBoxTree::compute_first_entity_collision(const Point& point,
                                                const Mesh& mesh) const
{
  // Check that tree has been built
  check_built();

  // Delegate call to implementation
  dolfin_assert(_tree);
  return _tree->compute_first_entity_collision(point, mesh);
}
//-----------------------------------------------------------------------------
std::pair<mesh_index, double>
BoundingBoxTree::compute_closest_entity(const Point& point,
                                        const Mesh& mesh) const
{
  // Check that tree has been built
  check_built();

  // Delegate call to implementation
  dolfin_assert(_tree);
  return _tree->compute_closest_entity(point, mesh);
}
//-----------------------------------------------------------------------------
std::pair<mesh_index, double>
BoundingBoxTree::compute_closest_point(const Point& point) const
{
  // Check that tree has been built
  check_built();

  // Delegate call to implementation
  dolfin_assert(_tree);
  return _tree->compute_closest_point(point);
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::check_built() const
{
  if (!_tree)
  {
    dolfin_error("BoundingBoxTree.cpp",
                 "compute collisions with bounding box tree",
                 "Bounding box tree has not been built. You need to call tree.build()");
  }
}
//-----------------------------------------------------------------------------
