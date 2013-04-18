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
// Last changed: 2013-04-18

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshGeometry.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Point.h>
#include "BoundingBoxTree.h"

using namespace dolfin;

// Helper function for getting bounding box
inline double* _get_bbox(std::vector<double>& bboxes,
                         unsigned int i, unsigned int gdim)
{
  dolfin_assert(2*gdim*(i + 1) <= bboxes.size());
  return bboxes.data() + 2*gdim*i;
}

// Helper function for getting bounding box (const version)
inline const double* _get_bbox(const std::vector<double>& bboxes,
                               unsigned int i, unsigned int gdim)
{
  dolfin_assert(2*gdim*(i + 1) <= bboxes.size());
  return bboxes.data() + 2*gdim*i;
}

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
    // Get bounding boxes
    const double* bbox_i = _get_bbox(bboxes, i, gdim);
    const double* bbox_j = _get_bbox(bboxes, j, gdim);

    // Compute midpoints
    const double xi = 0.5*(bbox_i[axis] + bbox_i[axis + gdim]);
    const double xj = 0.5*(bbox_j[axis] + bbox_j[axis + gdim]);

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
std::vector<unsigned int> BoundingBoxTree::find(const Point& point) const
{
  // Call recursive find function
  std::vector<unsigned int> entities;
  find(point.coordinates(), 0, entities);

  return entities;
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::build(const Mesh& mesh, unsigned int dimension)
{
  // Check dimension
  if (dimension < 1 or dimension > mesh.topology().dim())
  {
    dolfin_error("BoundingBoxTree.cpp",
                 "compute bounding box tree",
                 "dimension must be a number between 1 and %d",
                 mesh.topology().dim());
  }

  // Initialize entities of given dimension if they don't exist
  mesh.init(dimension);

  // Compute upper bound for size of bounding box tree
  const unsigned int num_leaves = mesh.num_entities(dimension);
  unsigned int n = 1;
  for (; n < num_leaves; n *= 2);
  const unsigned int N = 2*n; // could subtract 1 here...

  // Clear any old data
  bbox_tree.clear();
  bbox_entities.clear();
  bbox_coordinates.clear();

  // Allocate data for bounding box tree
  bbox_tree.resize(N);
  bbox_entities.resize(N);
  bbox_coordinates.resize(2*_gdim*N);
  for (unsigned int i = 0; i < N; ++i)
    bbox_tree[i] = -1;
  for (unsigned int i = 0; i < N; ++i)
    bbox_entities[i] = -1;
  for (unsigned int i = 0; i < 2*_gdim*N; ++i)
    bbox_coordinates[i] = 0.0;

  // Build bounding boxes for leaves
  std::vector<double> leaf_bboxes(2*_gdim*num_leaves);
  for (MeshEntityIterator it(mesh, dimension); !it.end(); ++it)
  {
    const unsigned int i = it->index();
    compute_bbox(_get_bbox(leaf_bboxes, i, _gdim), *it);
  }

  // Initialize leaf partition (to be sorted)
  std::vector<unsigned int> leaf_partition(num_leaves);
  for (unsigned int i = 0; i < num_leaves; ++i)
    leaf_partition[i] = i;

  // Recursively build the bounding box tree from the leaves
  unsigned int pos = 0;
  build(leaf_bboxes, leaf_partition, 0, num_leaves, 0, pos);

  info("Computed bounding box tree with %d nodes for %d entities.",
       pos / (2*_gdim), num_leaves);
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::build(const std::vector<double> leaf_bboxes,
                            std::vector<unsigned int> leaf_partition,
                            unsigned int begin,
                            unsigned int end,
                            unsigned int node,
                            unsigned int& pos)
{
  dolfin_assert(begin < end);

  // Add bounding box to tree. The pos variable is used to step
  // through the preallocated list of bounding box coordinates.
  bbox_tree[node] = pos;
  pos += 2*_gdim;

  // Compute bounding box
  double* bbox = get_bbox(node);
  compute_bbox(bbox, leaf_bboxes, leaf_partition, begin, end);

  // Reached leaf
  if (begin + 1 == end)
  {
    bbox_entities[node] = leaf_partition[begin];
    return;
  }

  // Compute longest axis of bounding box
  const unsigned int longest_axis = compute_longest_axis(bbox);

  // Sort boxes along axis
  std::sort(leaf_partition.begin() + begin, leaf_partition.begin() + end,
            ComparisonOperator(leaf_bboxes, _gdim, longest_axis));

  // Split boxes in two groups and call recursively
  const unsigned int pivot = (begin + end + 1) / 2;
  build(leaf_bboxes, leaf_partition, begin, pivot, 2*node + 1, pos);
  build(leaf_bboxes, leaf_partition, pivot, end,   2*node + 2, pos);
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
      xmin[j] = std::min(xmin[j], x[j]);
      xmax[j] = std::max(xmax[j], x[j]);
    }
  }
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::compute_bbox(double* bbox,
                                   const std::vector<double> bboxes,
                                   const std::vector<unsigned int> partition,
                                   unsigned int begin,
                                   unsigned int end) const
{
  // Get bounding box data
  double* xmin = bbox;
  double* xmax = bbox + _gdim;

  // Get coordinates for first box
  const double* _bbox = _get_bbox(bboxes, partition[begin], _gdim);
  for (unsigned int j = 0; j < _gdim; ++j)
  {
    xmin[j] = _bbox[j];
    xmax[j] = _bbox[j + _gdim];
  }

  // Compute min and max over remaining boxes
  for (unsigned int i = begin + 1; i < end; ++i)
  {
    const double* _bbox = _get_bbox(bboxes, partition[i], _gdim);
    for (unsigned int j = 0; j < _gdim; ++j)
    {
      xmin[j] = std::min(xmin[j], _bbox[j]);
      xmax[j] = std::max(xmax[j], _bbox[j + _gdim]);
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
    {
      longest_axis = j;
      longest_axis_length = axis_length;
    }
  }

  return longest_axis;
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::find(const double* x,
                           unsigned int node,
                           std::vector<unsigned int>& entities) const
{
  // Three cases: either the point is not contained (so skip branch),
  // or it's contained in a leaf (so add it) or it's contained in the
  // bounding box (so search the two children).

  if (!contains(x, node))
    return;
  else if (is_leaf(node))
    entities.push_back(bbox_entities[node]);
  else
  {
    find(x, 2*node + 1, entities);
    find(x, 2*node + 2, entities);
  }
}
//-----------------------------------------------------------------------------
