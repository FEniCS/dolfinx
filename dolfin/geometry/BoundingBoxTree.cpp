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
// Last changed: 2013-05-02

#include <dolfin/log/log.h>
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
  // FIXME: Add 1D, 2D

  // Select implementation
  if (!_tree)
    _tree.reset(new BoundingBoxTree3D());

  // Build tree
  dolfin_assert(_tree);
  _tree->build(mesh);
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::build(const Mesh& mesh, unsigned int dimension)
{
  // Select implementation
  _tree.reset(new BoundingBoxTree3D());

  // Build tree
  dolfin_assert(_tree);
  _tree->build(mesh, dimension);
}
//-----------------------------------------------------------------------------
std::vector<unsigned int> BoundingBoxTree::find(const Point& point) const
{
  dolfin_assert(_tree);
  return _tree->find(point);
}
//-----------------------------------------------------------------------------
