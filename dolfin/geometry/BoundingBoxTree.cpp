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
// Last changed: 2013-04-17

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshGeometry.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/Vertex.h>
#include "BoundingBoxTree.h"

using namespace dolfin;

// Comparison operator for sorting of bounding boxes
struct ComparisonOperator
{
  const std::vector<double>& bboxes;
  unsigned int gdim;
  unsigned int axis;

  ComparisonOperator(const std::vector<double>& bboxes,
                     unsigned int gdim,
                     unsigned int axis)
    : bboxes(bboxes), gdim(gdim), axis(axis) {}

  inline bool operator()(unsigned int i, unsigned int j)
  {
    // Compute midpoints
    const double xi = 0.5*(bboxes[2*gdim*i + axis] + bboxes[2*gdim*i + axis + gdim]);
    const double xj = 0.5*(bboxes[2*gdim*j + axis] + bboxes[2*gdim*j + axis + gdim]);

    // Compare
    return xi < xj;
  }

};

//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(const Mesh& mesh)
  : _gdim(mesh.geometry().dim())
{
  build(mesh, mesh.topology().dim());
}
//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(const Mesh& mesh, unsigned int dimension)
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
void BoundingBoxTree::build(const Mesh& mesh, unsigned int dimension)
{
  cout << "Building bounding box tree" << endl;

  // Check dimension
  if (dimension < 1 or dimension > mesh.topology().dim())
  {
    dolfin_error("BoundingBoxTree.cpp",
                 "compute bounding box tree",
                 "dimension must be a number between 1 and %d",
                 mesh.topology().dim());
  }

  // Allocate data for bounding box tree
  const unsigned int num_leaves = mesh.num_entities(dimension);
  bbox_tree.clear();
  bbox_tree.resize(2*_gdim*2*num_leaves - 1); // should be precisely enough

  // Build bounding boxes for leaves
  std::vector<double> leaf_bboxes(2*_gdim*num_leaves);
  for (MeshEntityIterator it(mesh, dimension); !it.end(); ++it)
    compute_bbox(leaf_bboxes.data() + 2*_gdim*it->index(), *it);

  // Initialize leaf partition (to be sorted)
  std::vector<unsigned int> leaf_partition(num_leaves);
  for (unsigned int i = 0; i < num_leaves; ++i)
    leaf_partition[i] = i;

  // Recursively build the bounding box tree from the leaves
  build(bbox_tree, leaf_partition, leaf_bboxes, 0, num_leaves, 0);
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::build(const std::vector<double> bbox_tree,
                            std::vector<unsigned int> leaf_partition,
                            const std::vector<double> leaf_bboxes,
                            unsigned int begin,
                            unsigned int end,
                            unsigned int node)
{
  cout << "Creating bounding box for set of " << (end - begin) << " leaves" << endl;
  cout << "node = " << node << endl;

  if (end < begin + 2)
  {
    cout << "Reached leaf" << endl;
    return;
  }

  // Compute bounding box
  double* bbox = get_bbox(node);
  compute_bbox(bbox, leaf_bboxes, begin, end);

  // Compute longest axis of bounding box
  const unsigned int longest_axis = compute_longest_axis(bbox);

  // Sort boxes along axis
  std::sort(leaf_partition.begin() + begin, leaf_partition.begin() + end,
            ComparisonOperator(leaf_bboxes, _gdim, longest_axis));

  // Split boxes in two groups and call recursively
  const unsigned int pivot = (begin + end) / 2;
  build(bbox_tree, leaf_partition, leaf_bboxes, begin, pivot, 2*node + 1);
  build(bbox_tree, leaf_partition, leaf_bboxes, pivot, end,   2*node + 2);
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
  for (unsigned int j = 0; j < _gdim; ++j)
    xmin[j] = xmax[j] = x[j];

  // Compute min and max over remaining vertices
  for (unsigned int i = 1; i < num_vertices; ++i)
  {
    const double* x = geometry.x(vertices[i]);
    for (unsigned int j = 0; j < _gdim; ++j)
    {
      xmin[j] = std::min(x[j], xmin[j]);
      xmax[j] = std::max(x[j], xmax[j]);
    }
  }
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::compute_bbox(double* bbox,
                                   const std::vector<double> bboxes,
                                   unsigned int begin,
                                   unsigned int end) const
{
  // Get bounding box data
  double* xmin = bbox;
  double* xmax = bbox + _gdim;

  // Get coordinates for first box
  const double* x = bboxes.data() + 2*_gdim*begin;
  for (unsigned int j = 0; j < _gdim; ++j)
  {
    xmin[j] = x[j];
    xmax[j] = x[j + _gdim];
  }

  // Compute min and max over remaining boxes
  for (unsigned int i = begin + 1; i < end; ++i)
  {
    const double* x = bboxes.data() + 2*_gdim*i;
    for (unsigned int j = 0; j < _gdim; ++j)
    {
      xmin[j] = std::min(x[j], xmin[j]);
      xmax[j] = std::max(x[j + _gdim], xmax[j]);
    }
  }
}
//-----------------------------------------------------------------------------
unsigned int BoundingBoxTree::compute_longest_axis(const double* bbox) const
{
  // Get bounding box data
  const double* xmin = bbox;
  const double* xmax = bbox + _gdim;

  // Check maximum axis
  unsigned int longest_axis = 0;
  double longest_axis_length = xmax[0] - xmin[0];
  for (unsigned int j = 1; j < _gdim; ++j)
  {
    const double axis_length = xmax[j] - xmin[j];
    if (axis_length > longest_axis_length)
      longest_axis = j;
  }

  cout << "Longest is axis " << longest_axis << ": " << longest_axis_length << endl;

  return longest_axis;
}
//-----------------------------------------------------------------------------
