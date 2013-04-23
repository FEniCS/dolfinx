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
// Last changed: 2013-04-23

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshGeometry.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Point.h>
#include "BoundingBoxTree.h"

using namespace dolfin;

// Comparison operators for sorting of bounding boxes. Boxes are
// sorted by their midpoints along the longest axis.

struct less_3d_x
{
  const std::vector<double>& bboxes;
  less_3d_x(const std::vector<double>& bboxes): bboxes(bboxes) {}

  inline bool operator()(unsigned int i, unsigned int j)
  {
    const double* bi = bboxes.data() + 6*i;
    const double* bj = bboxes.data() + 6*j;
    return (bi[0] + bi[3]) < (bj[0] + bj[3]);
  }
};

struct less_3d_y
{
  const std::vector<double>& bboxes;
  less_3d_y(const std::vector<double>& bboxes): bboxes(bboxes) {}

  inline bool operator()(unsigned int i, unsigned int j)
  {
    const double* bi = bboxes.data() + 6*i;
    const double* bj = bboxes.data() + 6*j;
    return (bi[1] + bi[4]) < (bj[1] + bj[4]);
  }
};

struct less_3d_z
{
  const std::vector<double>& bboxes;
  less_3d_z(const std::vector<double>& bboxes): bboxes(bboxes) {}

  inline bool operator()(unsigned int i, unsigned int j)
  {
    const double* bi = bboxes.data() + 6*i;
    const double* bj = bboxes.data() + 6*j;
    return (bi[2] + bi[5]) < (bj[2] + bj[5]);
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
  find(point.coordinates(), bboxes.size() - 1, entities);

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

  // Clear existing data if any
  bboxes.clear();

  // Initialize bounding boxes for leaves
  const unsigned int num_leaves = mesh.num_entities(dimension);
  std::vector<double> leaf_bboxes(2*_gdim*num_leaves);
  for (MeshEntityIterator it(mesh, dimension); !it.end(); ++it)
    compute_bbox_of_entity(leaf_bboxes.data() + 2*_gdim*it->index(), *it);

  // Initialize leaf partition (to be sorted)
  std::vector<unsigned int> leaf_partition(num_leaves);
  for (unsigned int i = 0; i < num_leaves; ++i)
    leaf_partition[i] = i;

  // Recursively build the bounding box tree from the leaves. We switch
  // on dimension at the highest possible level to maximize performance.
  switch (_gdim)
  {
  case 1:
    cout << "not implemented" << endl;
    break;
  case 2:
    cout << "not implemented" << endl;
    break;
  case 3:
    build_3d(leaf_bboxes, leaf_partition.begin(), leaf_partition.end(), 0);
    break;
  default:
    cout << "not implemented" << endl;
  }

  info("Computed bounding box tree with %d nodes for %d entities.",
       bboxes.size(), num_leaves);
}
//-----------------------------------------------------------------------------
unsigned int BoundingBoxTree::build_3d(const std::vector<double>& leaf_bboxes,
                                       const std::vector<unsigned int>::iterator& begin,
                                       const std::vector<unsigned int>::iterator& end,
                                       short unsigned int parent_axis)
{
  dolfin_assert(begin < end);

  // Create empty bounding box data
  BBox bbox;

  // Reached leaf
  if (end - begin == 1)
  {
    // Get bounding box coordinates for leaf
    const unsigned int i = *begin;
    const double* b = leaf_bboxes.data() + 6*i;

    // Store bounding box data
    bbox.entity = i;
    bbox.child_0 = 0;
    bbox.child_1 = 0;
    bbox.axis = parent_axis;
    bbox.min = b[parent_axis];
    bbox.max = b[parent_axis + 3];
    bboxes.push_back(bbox);

    return bboxes.size() - 1;
  }

  // Compute bounding box of all bounding boxes
  double _bbox[6];
  compute_bbox_of_bboxes_3d(_bbox, leaf_bboxes, begin, end);

  // Compute longest axis of bounding box
  bbox.axis = compute_longest_axis_3d(_bbox);
  bbox.min = _bbox[bbox.axis];
  bbox.max = _bbox[bbox.axis + 3];

  // Sort bounding boxes along longest axis
  std::vector<unsigned int>::iterator middle = begin + (end - begin) / 2;
  switch (bbox.axis)
  {
  case 0:
    std::nth_element(begin, middle, end, less_3d_x(leaf_bboxes));
    break;
  case 1:
    std::nth_element(begin, middle, end, less_3d_y(leaf_bboxes));
    break;
  default:
    std::nth_element(begin, middle, end, less_3d_z(leaf_bboxes));
  }

  // Split boxes in two groups and call recursively
  bbox.child_0 = build_3d(leaf_bboxes, begin, middle, bbox.axis);
  bbox.child_1 = build_3d(leaf_bboxes, middle, end,   bbox.axis);

  // Store bounding box data. Note that root box will be added last.
  bboxes.push_back(bbox);
  for (unsigned int i = 0; i < 6; i++)
    bbox_coordinates.push_back(_bbox[i]);

  return bboxes.size() - 1;
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::compute_bbox_of_entity(double* bbox,
                                             const MeshEntity& entity) const
{
  // Get bounding box coordinates
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
void BoundingBoxTree::
compute_bbox_of_bboxes_3d(double* bbox,
                          const std::vector<double>& leaf_bboxes,
                          const std::vector<unsigned int>::iterator& begin,
                          const std::vector<unsigned int>::iterator& end)
{
  typedef std::vector<unsigned int>::const_iterator iterator;

  // Get coordinates for first box
  iterator it = begin;
  const double* b = leaf_bboxes.data() + 6*(*it);
  bbox[0] = b[0];
  bbox[1] = b[1];
  bbox[2] = b[2];
  bbox[3] = b[3];
  bbox[4] = b[4];
  bbox[5] = b[5];

  // Compute min and max over remaining boxes
  for (; it != end; ++it)
  {
    const double* b = leaf_bboxes.data() + 6*(*it);
    if (b[0] < bbox[0]) bbox[0] = b[0];
    if (b[1] < bbox[1]) bbox[1] = b[1];
    if (b[2] < bbox[2]) bbox[2] = b[2];
    if (b[3] > bbox[3]) bbox[3] = b[3];
    if (b[4] > bbox[4]) bbox[4] = b[4];
    if (b[5] > bbox[5]) bbox[5] = b[5];
  }
}
//-----------------------------------------------------------------------------
short unsigned int BoundingBoxTree::
compute_longest_axis_3d(const double* bbox) const
{
  const double x = bbox[3] - bbox[0];
  const double y = bbox[4] - bbox[1];
  const double z = bbox[5] - bbox[2];

  if (x > y && x > z)
    return 0;
  else if (y > z)
    return 1;
  else
    return 2;
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::find(const double* x,
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
