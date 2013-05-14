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
// Last changed: 2013-05-15

#include <dolfin/log/log.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/mesh/Mesh.h>
#include "BoundingBoxTree1D.h"
#include "BoundingBoxTree2D.h"
#include "BoundingBoxTree3D.h"
#include "BoundingBoxTree.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(const Mesh& mesh)
  : _mesh(reference_to_no_delete_pointer(mesh)),
    _tdim(mesh.topology().dim())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(boost::shared_ptr<const Mesh> mesh)
  : _mesh(mesh),
    _tdim(mesh->topology().dim())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(const Mesh& mesh,
                                 unsigned int dim)
  : _mesh(reference_to_no_delete_pointer(mesh)),
    _tdim(dim)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(boost::shared_ptr<const Mesh> mesh,
                                 unsigned int dim)
  : _mesh(mesh),
    _tdim(dim)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundingBoxTree::~BoundingBoxTree()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::build()
{
  dolfin_assert(_mesh);

  // Select implementation
  switch (_mesh->geometry().dim())
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
                 _mesh->geometry().dim());
  }

  // Build tree
  dolfin_assert(_tree);
  _tree->build(*_mesh, _tdim);
}
//-----------------------------------------------------------------------------
std::vector<unsigned int>
BoundingBoxTree::compute_collisions(const Point& point) const
{
  // Check that tree has been built
  if (!_tree)
  {
    dolfin_error("BoundingBoxTree.cpp",
                 "compute collisions with bounding box tree",
                 "Bounding box tree has not been build. You need to call tree.build()");
  }

  // Delegate call to implementation
  dolfin_assert(_tree);
  return _tree->compute_collisions(point);
}
//-----------------------------------------------------------------------------
std::vector<unsigned int>
BoundingBoxTree::compute_entity_collisions(const Point& point) const
{
  // Check that tree has been built
  if (!_tree)
  {
    dolfin_error("BoundingBoxTree.cpp",
                 "compute collisions with bounding box tree",
                 "Bounding box tree has not been build. You need to call tree.build()");
  }

  // Delegate call to implementation
  dolfin_assert(_tree);
  return _tree->compute_entity_collisions(point);
}
//-----------------------------------------------------------------------------
unsigned int
BoundingBoxTree::compute_first_collision(const Point& point) const
{
  // Check that tree has been built
  if (!_tree)
  {
    dolfin_error("BoundingBoxTree.cpp",
                 "compute collisions with bounding box tree",
                 "Bounding box tree has not been build. You need to call tree.build()");
  }

  // Delegate call to implementation
  dolfin_assert(_tree);
  return _tree->compute_first_collision(point);
}
//-----------------------------------------------------------------------------
unsigned int
BoundingBoxTree::compute_first_entity_collision(const Point& point) const
{
  // Check that tree has been built
  if (!_tree)
  {
    dolfin_error("BoundingBoxTree.cpp",
                 "compute collisions with bounding box tree",
                 "Bounding box tree has not been build. You need to call tree.build()");
  }

  // Delegate call to implementation
  dolfin_assert(_tree);
  return _tree->compute_first_entity_collision(point);
}
//-----------------------------------------------------------------------------
