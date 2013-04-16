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
// Last changed: 2013-04-16

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshGeometry.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/Vertex.h>
#include "BoundingBoxTree.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(const Mesh& mesh)
  : _gdim(mesh.geometry().dim())
{
  build(mesh, mesh.topology().dim());
}
//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(const Mesh& mesh, std::size_t dimension)
  : _gdim(mesh.geometry().dim())
{
  build(mesh, dimension);
}
//-----------------------------------------------------------------------------
BoundingBoxTree::~BoundingBoxTree()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::build(const Mesh& mesh, std::size_t dimension)
{
  // Check dimension
  if (dimension < 1 or dimension > mesh.topology().dim())
  {
    dolfin_error("BoundingBoxTree.cpp",
                 "compute bounding box tree",
                 "dimension must be a number between 1 and %d",
                 mesh.topology().dim());
  }

  // Storage for leaf bounding boxes
  std::vector<double> bboxes(2*_gdim*mesh.num_entities(dimension));

  // Build leaf bounding boxes
  for (MeshEntityIterator it(mesh, dimension); !it.end(); ++it)
    compute_bbox(bboxes.data() + 2*_gdim*it->index(), *it);
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::compute_bbox(double* bbox,
                                   const MeshEntity& entity) const
{
  // Get bounding box data
  double* xmin = bbox;
  double* xmax = bbox + _gdim;

  // Get mesh entity data
  const MeshGeometry& geometry = entity.mesh().geometry();
  const size_t num_vertices = entity.num_entities(0);
  const unsigned int* vertices = entity.entities(0);
  dolfin_assert(num_vertices >= 2);

  // Get coordinates for first vertex
  const double* x = geometry.x(vertices[0]);
  for (std::size_t j = 0; j < _gdim; ++j)
    xmin[j] = xmax[j] = x[j];

  // Compute min and max over all other vertices
  for (std::size_t i = 1; i < num_vertices; ++i)
  {
    const double* x = geometry.x(vertices[i]);
    for (std::size_t j = 0; j < _gdim; ++j)
    {
      xmin[j] = std::min(x[j], xmin[j]);
      xmax[j] = std::max(x[j], xmax[j]);
    }
  }
}
//-----------------------------------------------------------------------------
