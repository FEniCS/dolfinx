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
// Last changed: 2013-05-28

// Define a maximum dimension used for a local array in the recursive
// build function. Speeds things up compared to allocating it in each
// recursion and is more convenient than sending it around.
#define MAX_DIM 6

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Point.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include "BoundingBoxTree1D.h" // used for internal point search tree
#include "BoundingBoxTree2D.h" // used for internal point search tree
#include "BoundingBoxTree3D.h" // used for internal point search tree
#include "GenericBoundingBoxTree.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericBoundingBoxTree::GenericBoundingBoxTree() : _tdim(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void GenericBoundingBoxTree::build(const Mesh& mesh, std::size_t tdim)
{
  // Check dimension
  if (tdim < 1 or tdim > mesh.topology().dim())
  {
    dolfin_error("GenericBoundingBoxTree.cpp",
                 "compute bounding box tree",
                 "Dimension must be a number between 1 and %d",
                 mesh.topology().dim());
  }

  // Clear existing data if any
  clear();

  // Store topological dimension (only used for checking that entity
  // collisions can only be computed with cells)
  _tdim = tdim;

  // Initialize entities of given dimension if they don't exist
  mesh.init(tdim);

  // Create bounding boxes for all entities (leaves)
  const std::size_t _gdim = gdim();
  const mesh_index num_leaves = mesh.num_entities(tdim);
  std::vector<double> leaf_bboxes(2*_gdim*num_leaves);
  for (MeshEntityIterator it(mesh, tdim); !it.end(); ++it)
    compute_bbox_of_entity(leaf_bboxes.data() + 2*_gdim*it->index(), *it, _gdim);

  // Create leaf partition (to be sorted)
  std::vector<mesh_index> leaf_partition(num_leaves);
  for (mesh_index i = 0; i < num_leaves; ++i)
    leaf_partition[i] = i;

  // Recursively build the bounding box tree from the leaves
  build(leaf_bboxes, leaf_partition.begin(), leaf_partition.end(), _gdim);

  info("Computed bounding box tree with %d nodes for %d entities.",
       _bboxes.size(), num_leaves);
}
//-----------------------------------------------------------------------------
void GenericBoundingBoxTree::build(const std::vector<Point>& points)
{
  // Clear existing data if any
  clear();

  // Create leaf partition (to be sorted)
  const mesh_index num_leaves = points.size();
  std::vector<mesh_index> leaf_partition(num_leaves);
  for (mesh_index i = 0; i < num_leaves; ++i)
    leaf_partition[i] = i;

  // Recursively build the bounding box tree from the leaves
  build(points, leaf_partition.begin(), leaf_partition.end(), gdim());

  info("Computed bounding box tree with %d nodes for %d points.",
       _bboxes.size(), num_leaves);
}
//-----------------------------------------------------------------------------
std::vector<mesh_index>
GenericBoundingBoxTree::compute_collisions(const Point& point) const
{
  // Call recursive find function
  std::vector<mesh_index> entities;
  compute_collisions(point, _bboxes.size() - 1, entities);

  return entities;
}
//-----------------------------------------------------------------------------
std::vector<mesh_index>
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
  std::vector<mesh_index> entities;
  compute_entity_collisions(point, _bboxes.size() - 1, entities, mesh);

  return entities;
}
//-----------------------------------------------------------------------------
mesh_index
GenericBoundingBoxTree::compute_first_collision(const Point& point) const
{
  // Call recursive find function
  return compute_first_collision(point, _bboxes.size() - 1);
}
//-----------------------------------------------------------------------------
mesh_index
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
std::pair<mesh_index, double>
GenericBoundingBoxTree::compute_closest_entity(const Point& point,
                                               const Mesh& mesh) const
{
  // Closest entity only implemented for cells. Consider extending this.
  if (_tdim != mesh.topology().dim())
  {
    dolfin_error("GenericBoundingBoxTree.cpp",
                 "compute closest entity of point",
                 "Closest-entity is only implemented for cells");
  }

  // Compute point search tree if not already done
  build_point_search_tree(mesh);

  // Search point cloud to get a good starting guess
  dolfin_assert(_point_search_tree);
  double r = _point_search_tree->compute_closest_point(point).second;

  // Initialize index and distance to closest entity
  mesh_index closest_entity = std::numeric_limits<mesh_index>::max();
  double R2 = r*r;

  // Call recursive find function
  compute_closest_entity(point, _bboxes.size() - 1, mesh, closest_entity, R2);

  // Sanity check
  dolfin_assert(closest_entity < std::numeric_limits<mesh_index>::max());

  std::pair<mesh_index, double> ret(closest_entity, sqrt(R2));
  return ret;
}
//-----------------------------------------------------------------------------
std::pair<mesh_index, double>
GenericBoundingBoxTree::compute_closest_point(const Point& point) const
{
  // Closest point only implemented for point cloud
  if (_tdim != 0)
  {
    dolfin_error("GenericBoundingBoxTree.cpp",
                 "compute closest point",
                 "Search tree has not been built for point cloud");
  }

  // Note that we don't compute a point search tree here... That would
  // be weird.

  // Get initial guess by picking the distance to a "random" point
  mesh_index closest_point = 0;
  double R2 = compute_squared_distance_point(point.coordinates(),
                                             closest_point);

  // Call recursive find function
  compute_closest_point(point, _bboxes.size() - 1, closest_point, R2);

  std::pair<mesh_index, double> ret(closest_point, sqrt(R2));
  return ret;
}
//-----------------------------------------------------------------------------
// Implementation of protected functions
//-----------------------------------------------------------------------------
void GenericBoundingBoxTree::clear()
{
  _tdim = 0;
  _bboxes.clear();
  _bbox_coordinates.clear();
  _point_search_tree.reset();
}
//-----------------------------------------------------------------------------
mesh_index
GenericBoundingBoxTree::build(const std::vector<double>& leaf_bboxes,
                              const std::vector<mesh_index>::iterator& begin,
                              const std::vector<mesh_index>::iterator& end,
                              std::size_t gdim)
{
  dolfin_assert(begin < end);

  // Create empty bounding box data
  BBox bbox;

  // Reached leaf
  if (end - begin == 1)
  {
    // Get bounding box coordinates for leaf
    const mesh_index entity_index = *begin;
    const double* b = leaf_bboxes.data() + 2*gdim*entity_index;

    // Store bounding box data
    bbox.child_0 = _bboxes.size(); // child_0 == node denotes a leaf
    bbox.child_1 = entity_index;   // index of entity contained in leaf
    return add_bbox(bbox, b, gdim);
  }

  // Compute bounding box of all bounding boxes
  double b[MAX_DIM];
  std::size_t axis;
  compute_bbox_of_bboxes(b, axis, leaf_bboxes, begin, end);

  // Sort bounding boxes along longest axis
  std::vector<mesh_index>::iterator middle = begin + (end - begin) / 2;
  sort_bboxes(axis, leaf_bboxes, begin, middle, end);

  // Split bounding boxes into two groups and call recursively
  bbox.child_0 = build(leaf_bboxes, begin, middle, gdim);
  bbox.child_1 = build(leaf_bboxes, middle, end, gdim);

  // Store bounding box data. Note that root box will be added last.
  return add_bbox(bbox, b, gdim);
}
//-----------------------------------------------------------------------------
mesh_index
GenericBoundingBoxTree::build(const std::vector<Point>& points,
                              const std::vector<mesh_index>::iterator& begin,
                              const std::vector<mesh_index>::iterator& end,
                              std::size_t gdim)
{
  dolfin_assert(begin < end);

  // Create empty bounding box data
  BBox bbox;

  // Reached leaf
  if (end - begin == 1)
  {
    // Store bounding box data
    const mesh_index point_index = *begin;
    bbox.child_0 = _bboxes.size(); // child_0 == node denotes a leaf
    bbox.child_1 = point_index;    // index of entity contained in leaf
    return add_point(bbox, points[point_index], gdim);
  }

  // Compute bounding box of all points
  double b[MAX_DIM];
  std::size_t axis;
  compute_bbox_of_points(b, axis, points, begin, end);

  // Sort bounding boxes along longest axis
  std::vector<mesh_index>::iterator middle = begin + (end - begin) / 2;
  sort_points(axis, points, begin, middle, end);

  // Split bounding boxes into two groups and call recursively
  bbox.child_0 = build(points, begin, middle, gdim);
  bbox.child_1 = build(points, middle, end, gdim);

  // Store bounding box data. Note that root box will be added last.
  return add_bbox(bbox, b, gdim);
}
//-----------------------------------------------------------------------------
void GenericBoundingBoxTree::build_point_search_tree(const Mesh& mesh) const
{
  // Don't build search tree if it already exists
  if (_point_search_tree)
    return;
  info("Building point search tree to accelerate distance queries.");

  // Create list of midpoints for all cells
  std::vector<Point> points;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    points.push_back(cell->midpoint());

  // Select implementation
  const std::size_t gdim = mesh.geometry().dim();
  switch (gdim)
  {
  case 1:
    _point_search_tree.reset(new BoundingBoxTree1D());
    break;
  case 2:
    _point_search_tree.reset(new BoundingBoxTree2D());
    break;
  case 3:
    _point_search_tree.reset(new BoundingBoxTree3D());
    break;
  default:
    dolfin_error("BoundingBoxTree.cpp",
                 "build bounding box tree",
                 "Not implemented for geometric dimension %d",
                 gdim);
  }

  // Build tree
  dolfin_assert(_point_search_tree);
  _point_search_tree->build(points);
}
//-----------------------------------------------------------------------------
void
GenericBoundingBoxTree::compute_collisions(const Point& point,
                                           mesh_index node,
                                           std::vector<mesh_index>& entities) const
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
                                                  mesh_index node,
                                                  std::vector<mesh_index>& entities,
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
    const mesh_index entity_index = bbox.child_1;
    Cell cell(mesh, entity_index);

    // Check entity
    if (cell.contains(point))
      entities.push_back(entity_index);
  }

  // Check both children
  else
  {
    compute_entity_collisions(point, bbox.child_0, entities, mesh);
    compute_entity_collisions(point, bbox.child_1, entities, mesh);
  }
}
//-----------------------------------------------------------------------------
mesh_index
GenericBoundingBoxTree::compute_first_collision(const Point& point,
                                                mesh_index node) const
{
  // Get max integer to signify not found
  mesh_index not_found = std::numeric_limits<mesh_index>::max();

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
    mesh_index c0 = compute_first_collision(point, bbox.child_0);
    if (c0 != not_found)
      return c0;

    // Check second child
    mesh_index c1 = compute_first_collision(point, bbox.child_1);
    if (c1 != not_found)
      return c1;
  }

  // Point not found
  return not_found;
}
//-----------------------------------------------------------------------------
mesh_index
GenericBoundingBoxTree::compute_first_entity_collision(const Point& point,
                                                       mesh_index node,
                                                       const Mesh& mesh) const
{
  // Get max integer to signify not found
  mesh_index not_found = std::numeric_limits<mesh_index>::max();

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
    const mesh_index entity_index = bbox.child_1;
    Cell cell(mesh, entity_index);

    // Check entity
    if (cell.contains(point))
      return entity_index;
  }

  // Check both children
  else
  {
    mesh_index c0 = compute_first_entity_collision(point, bbox.child_0, mesh);
    if (c0 != not_found)
      return c0;

    mesh_index c1 = compute_first_entity_collision(point, bbox.child_1, mesh);
    if (c1 != not_found)
      return c1;
  }

  // Point not found
  return not_found;
}
//-----------------------------------------------------------------------------
void
GenericBoundingBoxTree::compute_closest_entity(const Point& point,
                                               mesh_index node,
                                               const Mesh& mesh,
                                               mesh_index& closest_entity,
                                               double& R2) const
{
  // Get bounding box for current node
  const BBox& bbox = _bboxes[node];

  // If bounding box is outside radius, then don't search further
  const double r2 = compute_squared_distance_bbox(point.coordinates(), node);
  if (r2 > R2)
    return;

  // If box is leaf (which we know is inside radius), then shrink radius
  else if (is_leaf(bbox, node))
  {
    // Get entity (child_1 denotes entity index for leaves)
    dolfin_assert(_tdim == mesh.topology().dim());
    const mesh_index entity_index = bbox.child_1;
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
void
GenericBoundingBoxTree::compute_closest_point(const Point& point,
                                              mesh_index node,
                                              mesh_index& closest_point,
                                              double& R2) const
{
  // Get bounding box for current node
  const BBox& bbox = _bboxes[node];

  // If box is leaf, then compute distance and shrink radius
  if (is_leaf(bbox, node))
  {
    const double r2 = compute_squared_distance_point(point.coordinates(), node);
    if (r2 < R2)
    {
      closest_point = bbox.child_1;
      R2 = r2;
    }
  }
  else
  {
    // If bounding box is outside radius, then don't search further
    const double r2 = compute_squared_distance_bbox(point.coordinates(), node);
    if (r2 > R2)
      return;

    // Check both children
    compute_closest_point(point, bbox.child_0, closest_point, R2);
    compute_closest_point(point, bbox.child_1, closest_point, R2);
  }
}
//-----------------------------------------------------------------------------
void GenericBoundingBoxTree::compute_bbox_of_entity(double* b,
                                                    const MeshEntity& entity,
                                                    std::size_t gdim) const
{
  // Get bounding box coordinates
  double* xmin = b;
  double* xmax = b + gdim;

  // Get mesh entity data
  const MeshGeometry& geometry = entity.mesh().geometry();
  const size_t num_vertices = entity.num_entities(0);
  const mesh_index* vertices = entity.entities(0);
  dolfin_assert(num_vertices >= 2);

  // Get coordinates for first vertex
  const double* x = geometry.x(vertices[0]);
  for (std::size_t j = 0; j < gdim; ++j)
    xmin[j] = xmax[j] = x[j];

  // Compute min and max over remaining vertices
  for (mesh_index i = 1; i < num_vertices; ++i)
  {
    const double* x = geometry.x(vertices[i]);
    for (std::size_t j = 0; j < gdim; ++j)
    {
      xmin[j] = std::min(xmin[j], x[j]);
      xmax[j] = std::max(xmax[j], x[j]);
    }
  }
}
//-----------------------------------------------------------------------------
void
GenericBoundingBoxTree::sort_points(std::size_t axis,
                                    const std::vector<Point>& points,
                                    const std::vector<mesh_index>::iterator& begin,
                                    const std::vector<mesh_index>::iterator& middle,
                                    const std::vector<mesh_index>::iterator& end)
{
  switch (axis)
  {
  case 0:
    std::nth_element(begin, middle, end, less_x_point(points));
    break;
  case 1:
    std::nth_element(begin, middle, end, less_y_point(points));
    break;
  default:
    std::nth_element(begin, middle, end, less_z_point(points));
  }
}
//-----------------------------------------------------------------------------
