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
// First added:  2013-04-18
// Last changed: 2013-05-10

#include <dolfin/log/LogStream.h>
#include <dolfin/mesh/Cell.h>
#include "BoundingBoxTree.h"
#include "MeshPointIntersection.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshPointIntersection::MeshPointIntersection(const Mesh& mesh,
                                             const Point& point)
  : _tree(mesh)
{
  // Build bounding box tree
  _tree.build();

  // Compute intersection
  compute_intersection(point);
}
//-----------------------------------------------------------------------------
MeshPointIntersection::MeshPointIntersection(boost::shared_ptr<const Mesh> mesh,
                                             const Point& point)
  : _tree(mesh)
{
  // Build bounding box tree
  _tree.build();

  // Compute intersection
  compute_intersection(point);
}
//-----------------------------------------------------------------------------
MeshPointIntersection::~MeshPointIntersection()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MeshPointIntersection::update(const Point& point)
{
  _intersected_cells.clear();
  compute_intersection(point);
}
//-----------------------------------------------------------------------------
void MeshPointIntersection::compute_intersection(const Point& point)
{
  dolfin_assert(_intersected_cells.size() == 0);

  // Compute list of candidates for intersection
  std::vector<unsigned int> cell_candidates = _tree.find(point);

  // FIXME: This should be moved to the BoundingBoxTree class

  // Extract subset of intersecting cells
  for (unsigned int i = 0; i < cell_candidates.size(); ++i)
  {
    const unsigned int cell_index = cell_candidates[i];
    Cell cell(*_tree._mesh, cell_index);
    if (cell.contains(point))
      _intersected_cells.push_back(cell_index);
  }
}
//-----------------------------------------------------------------------------
