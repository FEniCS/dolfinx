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
// Last changed: 2013-05-22

// Define a maximum dimension used for a local array in the recursive
// build function. Speeds things up compared to allocating it in each
// recursion and is more convenient than sending it around.
#define MAX_DIM 6

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Point.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include "GenericBoundingBoxTree.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericBoundingBoxTree::GenericBoundingBoxTree() : _tdim(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void GenericBoundingBoxTree::build(const Mesh& mesh, unsigned int tdim)
{
  // Check dimension
  if (tdim < 1 or tdim > mesh.topology().dim())
  {
    dolfin_error("GenericBoundingBoxTree.cpp",
                 "compute bounding box tree",
                 "Dimension must be a number between 1 and %d",
                 mesh.topology().dim());
  }

  // Store topological dimension
  _tdim = tdim;

  // Initialize entities of given dimension if they don't exist
  mesh.init(tdim);

  // Clear existing data if any
  _bboxes.clear();

  // Create bounding boxes for all entities (leaves)
  const unsigned int gdim = mesh.geometry().dim();
  const unsigned int num_leaves = mesh.num_entities(tdim);
  std::vector<double> leaf_bboxes(2*gdim*num_leaves);
  for (MeshEntityIterator it(mesh, tdim); !it.end(); ++it)
    compute_bbox_of_entity(leaf_bboxes.data() + 2*gdim*it->index(), *it, gdim);

  // Create leaf partition (to be sorted)
  std::vector<unsigned int> leaf_partition(num_leaves);
  for (unsigned int i = 0; i < num_leaves; ++i)
    leaf_partition[i] = i;

  // Recursively build the bounding box tree from the leaves
  build(leaf_bboxes, leaf_partition.begin(), leaf_partition.end(), gdim);

  info("Computed bounding box tree with %d nodes for %d entities.",
       _bboxes.size(), num_leaves);
}
//-----------------------------------------------------------------------------
unsigned int
GenericBoundingBoxTree::build(std::vector<double>& leaf_bboxes,
                              const std::vector<unsigned int>::iterator& begin,
                              const std::vector<unsigned int>::iterator& end,
                              unsigned int gdim)
{
  dolfin_assert(begin < end);

  // Create empty bounding box data
  BBox bbox;

  // Reached leaf
  if (end - begin == 1)
  {
    // Get bounding box coordinates for leaf
    const unsigned int entity_index = *begin;
    const double* b = leaf_bboxes.data() + 2*gdim*entity_index;

    // Store bounding box data
    bbox.child_0 = _bboxes.size(); // child_0 == node denotes a leaf
    bbox.child_1 = entity_index;   // index of entity contained in leaf
    return add_bbox(bbox, b, gdim);
  }

  // Compute bounding box of all bounding boxes
  double b[MAX_DIM];
  short unsigned int axis;
  compute_bbox_of_bboxes(b, axis, leaf_bboxes, begin, end);

  // Sort bounding boxes along longest axis
  std::vector<unsigned int>::iterator middle = begin + (end - begin) / 2;
  sort_bboxes(axis, leaf_bboxes, begin, middle, end);

  // Split bounding boxes into two groups and call recursively
  bbox.child_0 = build(leaf_bboxes, begin, middle, gdim);
  bbox.child_1 = build(leaf_bboxes, middle, end, gdim);

  // Store bounding box data. Note that root box will be added last.
  return add_bbox(bbox, b, gdim);
}
//-----------------------------------------------------------------------------
std::vector<unsigned int>
GenericBoundingBoxTree::compute_collisions(const Point& point) const
{
  // Call recursive find function
  std::vector<unsigned int> entities;
  compute_collisions(point, _bboxes.size() - 1, entities);

  return entities;
}
//-----------------------------------------------------------------------------
std::vector<unsigned int>
GenericBoundingBoxTree::compute_entity_collisions(const Point& point,
                                                  const Mesh& mesh) const
{
  // Point in entity only implemented for cells. Consider extending this.
  if (_tdim != mesh.topology().dim())
  {
    dolfin_error("GenericBoundingBoxTree.cpp",
                 "compute collision between point and mesh entities",
                 "Point-in-entity is only implemented for cells");
  }

  // Call recursive find function
  std::vector<unsigned int> entities;
  compute_entity_collisions(point, _bboxes.size() - 1, entities, mesh);

  return entities;
}
//-----------------------------------------------------------------------------
unsigned int
GenericBoundingBoxTree::compute_first_collision(const Point& point) const
{
  // Call recursive find function
  return compute_first_collision(point, _bboxes.size() - 1);
}
//-----------------------------------------------------------------------------
unsigned int
GenericBoundingBoxTree::compute_first_entity_collision(const Point& point,
                                                       const Mesh& mesh) const
{
  // Point in entity only implemented for cells. Consider extending this.
  if (_tdim != mesh.topology().dim())
  {
    dolfin_error("GenericBoundingBoxTree.cpp",
                 "compute collision between point and mesh entities",
                 "Point-in-entity is only implemented for cells");
  }

  // Call recursive find function
  return compute_first_entity_collision(point, _bboxes.size() - 1, mesh);
}
//-----------------------------------------------------------------------------
std::pair<unsigned int, double>
GenericBoundingBoxTree::compute_closest_entity(const Point& point,
                                               const Mesh& mesh) const
{
  // Closest entity only implemented for cells. Consider extending this.
  if (_tdim != mesh.topology().dim())
  {
    dolfin_error("GenericBoundingBoxTree.cpp",
                 "compute collision between point and mesh entities",
                 "Closest-entity is only implemented for cells");
  }

  // Note: This algorithm may be optimized by finding a better
  // starting guess for R2. One option would be to store a small
  // number of bounding boxes (all boxes at a certain depth) and do a
  // linear search among those boxes to find a suitable value of R2
  // which partitions the boxes in two groups of equal size. One might
  // also consider building an additional search tree like in CGAL
  // when the function accelerate_distance_queries is called.

  // Initialize index and distance to closest entity
  unsigned int closest_entity = std::numeric_limits<unsigned int>::max();
  double R2 = std::numeric_limits<double>::max();

  // Call recursive find function
  compute_closest_entity(point, _bboxes.size() - 1, mesh, closest_entity, R2);

  // Sanity check
  dolfin_assert(closest_entity < std::numeric_limits<unsigned int>::max());

  std::pair<unsigned int, double> ret(closest_entity, R2);
  return ret;
}
//-----------------------------------------------------------------------------
void
GenericBoundingBoxTree::compute_collisions(const Point& point,
                                           unsigned int node,
                                           std::vector<unsigned int>& entities) const
{
  // Get bounding box for current node
  const BBox& bbox = _bboxes[node];

  // If point is not in bounding box, then don't search further
  if (!point_in_bbox(point.coordinates(), node))
    return;

  // If box is a leaf (which we know contains the point), then add it
  else if (is_leaf(bbox, node))
    entities.push_back(bbox.child_1); // child_1 denotes entity for leaves

  // Check both children
  else
  {
    compute_collisions(point, bbox.child_0, entities);
    compute_collisions(point, bbox.child_1, entities);
  }
}
//-----------------------------------------------------------------------------
void
GenericBoundingBoxTree::compute_entity_collisions(const Point& point,
                                                  unsigned int node,
                                                  std::vector<unsigned int>& entities,
                                                  const Mesh& mesh) const
{
  // Get bounding box for current node
  const BBox& bbox = _bboxes[node];

  // If point is not in bounding box, then don't search further
  if (!point_in_bbox(point.coordinates(), node))
    return;

  // If box is a leaf (which we know contains the point), then check entity
  else if (is_leaf(bbox, node))
  {
    // Get entity (child_1 denotes entity index for leaves)
    dolfin_assert(_tdim == mesh.topology().dim());
    const unsigned int entity_index = bbox.child_1;
    Cell cell(mesh, entity_index);

    // Check entity
    if (cell.contains(point))
      entities.push_back(entity_index);
  }

  // Check both children
  else
  {
    compute_collisions(point, bbox.child_0, entities);
    compute_collisions(point, bbox.child_1, entities);
  }
}
//-----------------------------------------------------------------------------
unsigned int
GenericBoundingBoxTree::compute_first_collision(const Point& point,
                                                unsigned int node) const
{
  // Get max integer to signify not found
  unsigned int not_found = std::numeric_limits<unsigned int>::max();

  // Get bounding box for current node
  const BBox& bbox = _bboxes[node];

  // If point is not in bounding box, then don't search further
  if (!point_in_bbox(point.coordinates(), node))
    return not_found;

  // If box is a leaf (which we know contains the point), then return it
  else if (is_leaf(bbox, node))
    return bbox.child_1; // child_1 denotes entity for leaves

  // Check both children
  else
  {
    unsigned int c0 = compute_first_collision(point, bbox.child_0);
    if (c0 != not_found)
      return c0;

    // Check second child
    unsigned int c1 = compute_first_collision(point, bbox.child_1);
    if (c1 != not_found)
      return c1;
  }

  // Point not found
  return not_found;
}
//-----------------------------------------------------------------------------
unsigned int
GenericBoundingBoxTree::compute_first_entity_collision(const Point& point,
                                                       unsigned int node,
                                                       const Mesh& mesh) const
{
  // Get max integer to signify not found
  unsigned int not_found = std::numeric_limits<unsigned int>::max();

  // Get bounding box for current node
  const BBox& bbox = _bboxes[node];

  // If point is not in bounding box, then don't search further
  if (!point_in_bbox(point.coordinates(), node))
    return not_found;

  // If box is a leaf (which we know contains the point), then check entity
  else if (is_leaf(bbox, node))
  {
    // Get entity (child_1 denotes entity index for leaves)
    dolfin_assert(_tdim == mesh.topology().dim());
    const unsigned int entity_index = bbox.child_1;
    Cell cell(mesh, entity_index);

    // Check entity
    if (cell.contains(point))
      return entity_index;
  }

  // Check both children
  else
  {
    unsigned int c0 = compute_first_collision(point, bbox.child_0);
    if (c0 != not_found)
      return c0;

    unsigned int c1 = compute_first_collision(point, bbox.child_1);
    if (c1 != not_found)
      return c1;
  }

  // Point not found
  return not_found;
}
//-----------------------------------------------------------------------------
void
GenericBoundingBoxTree::compute_closest_entity(const Point& point,
                                               unsigned int node,
                                               const Mesh& mesh,
                                               unsigned int& closest_entity,
                                               double& R2) const
{
  // Get bounding box for current node
  const BBox& bbox = _bboxes[node];

  // If bounding box is outside radius, then don't search further
  const double r2 = compute_squared_distance(point.coordinates(), node);
  if (r2 > R2)
    return;

  // If box is a leaf (which we know is inside the radius), then check entity
  else if (is_leaf(bbox, node))
  {
    // Get entity (child_1 denotes entity index for leaves)
    dolfin_assert(_tdim == mesh.topology().dim());
    const unsigned int entity_index = bbox.child_1;
    Cell cell(mesh, entity_index);

    // If entity is closer than best result so far, then return it
    const double r2 = cell.squared_distance(point);
    if (r2 < R2)
    {
      closest_entity = entity_index;
      R2 = r2;
    }
  }

  // Check both children
  else
  {
    compute_closest_entity(point, bbox.child_0, mesh, closest_entity, R2);
    compute_closest_entity(point, bbox.child_1, mesh, closest_entity, R2);
  }
}
//-----------------------------------------------------------------------------
void GenericBoundingBoxTree::compute_bbox_of_entity(double* b,
                                                    const MeshEntity& entity,
                                                    unsigned int gdim) const
{
  // Get bounding box coordinates
  double* xmin = b;
  double* xmax = b + gdim;

  // Get mesh entity data
  const MeshGeometry& geometry = entity.mesh().geometry();
  const size_t num_vertices = entity.num_entities(0);
  const unsigned int* vertices = entity.entities(0);
  dolfin_assert(num_vertices >= 2);

  // Get coordinates for first vertex
  const double* x = geometry.x(vertices[0]);
  for (unsigned int j = 0; j < gdim; ++j)
    xmin[j] = xmax[j] = x[j];

  // Compute min and max over remaining vertices
  for (unsigned int i = 1; i < num_vertices; ++i)
  {
    const double* x = geometry.x(vertices[i]);
    for (unsigned int j = 0; j < gdim; ++j)
    {
      xmin[j] = std::min(xmin[j], x[j]);
      xmax[j] = std::max(xmax[j], x[j]);
    }
  }
}
//-----------------------------------------------------------------------------
