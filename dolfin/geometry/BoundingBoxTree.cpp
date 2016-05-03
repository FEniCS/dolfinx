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
// Last changed: 2014-05-12

#include <dolfin/common/NoDeleter.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include "BoundingBoxTree1D.h"
#include "BoundingBoxTree2D.h"
#include "BoundingBoxTree3D.h"
#include "BoundingBoxTree.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree() : _mesh(0)
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

  _tree = GenericBoundingBoxTree::create(mesh.geometry().dim());

  // Build tree
  dolfin_assert(_tree);
  _tree->build(mesh, tdim);

  // Store mesh
  _mesh = &mesh;
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
std::vector<unsigned int>
BoundingBoxTree::compute_collisions(const Point& point) const
{
  // Check that tree has been built
  _check_built();

  // Delegate call to implementation
  dolfin_assert(_tree);
  return _tree->compute_collisions(point);
}
//-----------------------------------------------------------------------------
std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
  BoundingBoxTree::compute_collisions(const BoundingBoxTree& tree) const
{
  // Check that tree has been built
  _check_built();

  // Delegate call to implementation
  dolfin_assert(_tree);
  dolfin_assert(tree._tree);
  return _tree->compute_collisions(*tree._tree);
}
//-----------------------------------------------------------------------------
std::vector<unsigned int>
BoundingBoxTree::compute_entity_collisions(const Point& point) const
{
  // Check that tree has been built
  _check_built();

  // Delegate call to implementation
  dolfin_assert(_tree);
  dolfin_assert(_mesh);
  return _tree->compute_entity_collisions(point, *_mesh);
}
//-----------------------------------------------------------------------------
std::vector<unsigned int>
BoundingBoxTree::compute_process_collisions(const Point& point) const
{
  // Check that tree has been built
  _check_built();

  // Delegate call to implementation
  dolfin_assert(_tree);
  return _tree->compute_process_collisions(point);
}
//-----------------------------------------------------------------------------
std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
  BoundingBoxTree::compute_entity_collisions(const BoundingBoxTree& tree) const
{
  // Check that tree has been built
  _check_built();

  // Delegate call to implementation
  dolfin_assert(_tree);
  dolfin_assert(tree._tree);
  dolfin_assert(_mesh);
  dolfin_assert(tree._mesh);
  return _tree->compute_entity_collisions(*tree._tree, *_mesh, *tree._mesh);
}
//-----------------------------------------------------------------------------
unsigned int
BoundingBoxTree::compute_first_collision(const Point& point) const
{
  // Check that tree has been built
  _check_built();

  // Delegate call to implementation
  dolfin_assert(_tree);
  return _tree->compute_first_collision(point);
}
//-----------------------------------------------------------------------------
unsigned int
BoundingBoxTree::compute_first_entity_collision(const Point& point) const
{
  // Check that tree has been built
  _check_built();

  // Delegate call to implementation
  dolfin_assert(_tree);
  dolfin_assert(_mesh);
  return _tree->compute_first_entity_collision(point, *_mesh);
}
//-----------------------------------------------------------------------------
std::pair<unsigned int, double>
BoundingBoxTree::compute_closest_entity(const Point& point) const
{
  // Check that tree has been built
  _check_built();

  // Delegate call to implementation
  dolfin_assert(_tree);
  dolfin_assert(_mesh);
  return _tree->compute_closest_entity(point, *_mesh);
}
//-----------------------------------------------------------------------------
std::pair<unsigned int, double>
BoundingBoxTree::compute_closest_point(const Point& point) const
{
  // Check that tree has been built
  _check_built();

  // Delegate call to implementation
  dolfin_assert(_tree);
  return _tree->compute_closest_point(point);
}
//-----------------------------------------------------------------------------
bool BoundingBoxTree::collides(const Point& point) const
{
  return compute_first_collision(point) != std::numeric_limits<unsigned int>::max();
}
//-----------------------------------------------------------------------------
bool BoundingBoxTree::collides_entity(const Point& point) const
{
  return compute_first_entity_collision(point) != std::numeric_limits<unsigned int>::max();
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::_check_built() const
{
  if (!_tree)
  {
    dolfin_error("BoundingBoxTree.cpp",
                 "compute collisions with bounding box tree",
                 "Bounding box tree has not been built. You need to call tree.build()");
  }
}
//-----------------------------------------------------------------------------
