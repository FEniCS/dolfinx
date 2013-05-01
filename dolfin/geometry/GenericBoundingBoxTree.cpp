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
// First added:  2013-05-02
// Last changed: 2013-05-02

#include <dolfin/mesh/Point.h>
#include "GenericBoundingBoxTree.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::vector<unsigned int> GenericBoundingBoxTree::find(const Point& point) const
{
  // Call recursive find function
  std::vector<unsigned int> entities;
  find(point.coordinates(), bboxes.size() - 1, entities);

  return entities;
}
//-----------------------------------------------------------------------------
void GenericBoundingBoxTree::find(const double* x,
                                  unsigned int node,
                                  std::vector<unsigned int>& entities) const
{
  // Three cases: either the point is not contained (so skip branch),
  // or it's contained in a leaf (so add it) or it's contained in the
  // bounding box (so search the two children).

  const BBox& bbox = bboxes[node];

  if (!bbox.contains(x))
    return;
  else if (bbox.is_leaf())
    entities.push_back(bbox.entity);
  else
  {
    find(x, bbox.child_0, entities);
    find(x, bbox.child_1, entities);
  }
}
//-----------------------------------------------------------------------------
