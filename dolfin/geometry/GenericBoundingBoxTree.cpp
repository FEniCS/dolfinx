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
// Last changed: 2014-02-06

// Define a maximum dimension used for a local array in the recursive
// build function. Speeds things up compared to allocating it in each
// recursion and is more convenient than sending it around.
#define MAX_DIM 6

#include <dolfin/common/MPI.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
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
std::shared_ptr<GenericBoundingBoxTree>
GenericBoundingBoxTree::create(unsigned int gdim)
{
  std::shared_ptr<GenericBoundingBoxTree> tree;

  switch (gdim)
  {
  case 1:
    tree.reset(new BoundingBoxTree1D());
    break;
  case 2:
    tree.reset(new BoundingBoxTree2D());
    break;
  case 3:
    tree.reset(new BoundingBoxTree3D());
    break;
  default:
    dolfin_error("BoundingBoxTree.cpp",
                 "build bounding box tree",
                 "Not implemented for geometric dimension %d",
                 gdim);
  }
  return tree;
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
  const unsigned int num_leaves = mesh.num_entities(tdim);
  std::vector<double> leaf_bboxes(2*_gdim*num_leaves);
  for (MeshEntityIterator it(mesh, tdim); !it.end(); ++it)
    compute_bbox_of_entity(leaf_bboxes.data() + 2*_gdim*it->index(), *it, _gdim);

  // Create leaf partition (to be sorted)
  std::vector<unsigned int> leaf_partition(num_leaves);
  for (unsigned int i = 0; i < num_leaves; ++i)
    leaf_partition[i] = i;

  // Recursively build the bounding box tree from the leaves
  _build(leaf_bboxes, leaf_partition.begin(), leaf_partition.end(), _gdim);

  log(PROGRESS,
      "Computed bounding box tree with %d nodes for %d entities.",
      num_bboxes(), num_leaves);

  const std::size_t mpi_size = MPI::size(mesh.mpi_comm());
  if (mpi_size > 1)
  {
    // Send root node coordinates to all processes
    std::vector<double> send_bbox(_bbox_coordinates.end() - _gdim*2,
                                  _bbox_coordinates.end());
    std::vector<double> recv_bbox;
    MPI::all_gather(mesh.mpi_comm(), send_bbox, recv_bbox);
    std::vector<unsigned int> global_leaves(mpi_size);
    for (std::size_t i = 0; i != mpi_size; ++i)
      global_leaves[i] = i;

    _global_tree = create(_gdim);
    _global_tree->_build(recv_bbox,
                         global_leaves.begin(), global_leaves.end(), _gdim);

    info("Computed global bounding box tree with %d boxes.",
         _global_tree->num_bboxes());
    // Print on rank 0
    //    if(MPI::rank(mesh.mpi_comm()) == 0)
    //      std::cout << _global_tree->str() << "\n";

  }
}
//-----------------------------------------------------------------------------
void GenericBoundingBoxTree::build(const std::vector<Point>& points)
{
  // Clear existing data if any
  clear();

  // Create leaf partition (to be sorted)
  const unsigned int num_leaves = points.size();
  std::vector<unsigned int> leaf_partition(num_leaves);
  for (unsigned int i = 0; i < num_leaves; ++i)
    leaf_partition[i] = i;

  // Recursively build the bounding box tree from the leaves
  _build(points, leaf_partition.begin(), leaf_partition.end(), gdim());

  info("Computed bounding box tree with %d nodes for %d points.",
       num_bboxes(), num_leaves);
}
//-----------------------------------------------------------------------------
std::vector<unsigned int>
GenericBoundingBoxTree::compute_collisions(const Point& point) const
{
  // Call recursive find function
  std::vector<unsigned int> entities;
  _compute_collisions(*this, point, num_bboxes() - 1, entities, 0);

  return entities;
}
//-----------------------------------------------------------------------------
std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
  GenericBoundingBoxTree::compute_collisions(
    const GenericBoundingBoxTree& tree) const
{
  // Introduce new variables for clarity
  const GenericBoundingBoxTree& A(*this);
  const GenericBoundingBoxTree& B(tree);

  // Create data structures for storing collisions
  std::vector<unsigned int> entities_A;
  std::vector<unsigned int> entities_B;

  // Call recursive find function
  _compute_collisions(A, B,
                      A.num_bboxes() - 1, B.num_bboxes() - 1,
                      entities_A, entities_B, 0, 0);

  return std::make_pair(entities_A, entities_B);
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

  // Call recursive find function to compute bounding box candidates
  std::vector<unsigned int> entities;
  _compute_collisions(*this, point, num_bboxes() - 1, entities, &mesh);

  return entities;
}
//-----------------------------------------------------------------------------
std::vector<unsigned int>
GenericBoundingBoxTree::compute_process_collisions(const Point& point) const
{
  if (_global_tree)
    return _global_tree->compute_collisions(point);

  return std::vector<unsigned int>(1, 0);
}
//-----------------------------------------------------------------------------
std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
GenericBoundingBoxTree::compute_entity_collisions(
  const GenericBoundingBoxTree& tree,
  const Mesh& mesh_A,
  const Mesh& mesh_B) const
{
  // Introduce new variables for clarity
  const GenericBoundingBoxTree& A(*this);
  const GenericBoundingBoxTree& B(tree);

  // Create data structures for storing collisions
  std::vector<unsigned int> entities_A;
  std::vector<unsigned int> entities_B;

  // Call recursive find function
  _compute_collisions(A, B,
                      A.num_bboxes() - 1, B.num_bboxes() - 1,
                      entities_A, entities_B, &mesh_A, &mesh_B);

  return std::make_pair(entities_A, entities_B);
}
//-----------------------------------------------------------------------------
unsigned int
GenericBoundingBoxTree::compute_first_collision(const Point& point) const
{
  // Call recursive find function
  return _compute_first_collision(*this, point, num_bboxes() - 1);
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
  return _compute_first_entity_collision(*this, point, num_bboxes() - 1, mesh);
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
                 "compute closest entity of point",
                 "Closest-entity is only implemented for cells");
  }

  // Compute point search tree if not already done
  build_point_search_tree(mesh);

  // Search point cloud to get a good starting guess
  dolfin_assert(_point_search_tree);
  double r = _point_search_tree->compute_closest_point(point).second;

  // Initialize index and distance to closest entity
  unsigned int closest_entity = std::numeric_limits<unsigned int>::max();
  double R2 = r*r;

  // Call recursive find function
  _compute_closest_entity(*this, point, num_bboxes() - 1,
                          mesh, closest_entity, R2);

  // Sanity check
  dolfin_assert(closest_entity < std::numeric_limits<unsigned int>::max());

  std::pair<unsigned int, double> ret(closest_entity, sqrt(R2));
  return ret;
}
//-----------------------------------------------------------------------------
std::pair<unsigned int, double>
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
  unsigned int closest_point = 0;
  double R2 = compute_squared_distance_point(point.coordinates(),
                                             closest_point);

  // Call recursive find function
  _compute_closest_point(*this, point, num_bboxes() - 1, closest_point, R2);

  std::pair<unsigned int, double> ret(closest_point, sqrt(R2));
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
unsigned int
GenericBoundingBoxTree::_build(const std::vector<double>& leaf_bboxes,
                               const std::vector<unsigned int>::iterator& begin,
                               const std::vector<unsigned int>::iterator& end,
                               std::size_t gdim)
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
    bbox.child_0 = num_bboxes(); // child_0 == node denotes a leaf
    bbox.child_1 = entity_index; // index of entity contained in leaf
    return add_bbox(bbox, b, gdim);
  }

  // Compute bounding box of all bounding boxes
  double b[MAX_DIM];
  std::size_t axis;
  compute_bbox_of_bboxes(b, axis, leaf_bboxes, begin, end);

  // Sort bounding boxes along longest axis
  std::vector<unsigned int>::iterator middle = begin + (end - begin) / 2;
  sort_bboxes(axis, leaf_bboxes, begin, middle, end);

  // Split bounding boxes into two groups and call recursively
  bbox.child_0 = _build(leaf_bboxes, begin, middle, gdim);
  bbox.child_1 = _build(leaf_bboxes, middle, end, gdim);

  // Store bounding box data. Note that root box will be added last.
  return add_bbox(bbox, b, gdim);
}
//-----------------------------------------------------------------------------
unsigned int
GenericBoundingBoxTree::_build(const std::vector<Point>& points,
                               const std::vector<unsigned int>::iterator& begin,
                               const std::vector<unsigned int>::iterator& end,
                               std::size_t gdim)
{
  dolfin_assert(begin < end);

  // Create empty bounding box data
  BBox bbox;

  // Reached leaf
  if (end - begin == 1)
  {
    // Store bounding box data
    const unsigned int point_index = *begin;
    bbox.child_0 = num_bboxes(); // child_0 == node denotes a leaf
    bbox.child_1 = point_index;  // index of entity contained in leaf
    return add_point(bbox, points[point_index], gdim);
  }

  // Compute bounding box of all points
  double b[MAX_DIM];
  std::size_t axis;
  compute_bbox_of_points(b, axis, points, begin, end);

  // Sort bounding boxes along longest axis
  std::vector<unsigned int>::iterator middle = begin + (end - begin) / 2;
  sort_points(axis, points, begin, middle, end);

  // Split bounding boxes into two groups and call recursively
  bbox.child_0 = _build(points, begin, middle, gdim);
  bbox.child_1 = _build(points, middle, end, gdim);

  // Store bounding box data. Note that root box will be added last.
  return add_bbox(bbox, b, gdim);
}
//-----------------------------------------------------------------------------
void
GenericBoundingBoxTree::_compute_collisions(const GenericBoundingBoxTree& tree,
                                            const Point& point,
                                            unsigned int node,
                                            std::vector<unsigned int>& entities,
                                            const Mesh* mesh)
{
  // Get bounding box for current node
  const BBox& bbox = tree.get_bbox(node);

  // If point is not in bounding box, then don't search further
  if (!tree.point_in_bbox(point.coordinates(), node))
    return;

  // If box is a leaf (which we know contains the point), then add it
  else if (tree.is_leaf(bbox, node))
  {
    // child_1 denotes entity for leaves
    const unsigned int entity_index = bbox.child_1;

    // If we have a mesh, check that the candidate is really a collision
    if (mesh)
    {
      // Get cell
      Cell cell(*mesh, entity_index);
      if (cell.collides(point))
        entities.push_back(entity_index);
    }

    // Otherwise, add the candidate
    else
      entities.push_back(entity_index);
  }

  // Check both children
  else
  {
    _compute_collisions(tree, point, bbox.child_0, entities, mesh);
    _compute_collisions(tree, point, bbox.child_1, entities, mesh);
  }
}
//-----------------------------------------------------------------------------
void
GenericBoundingBoxTree::_compute_collisions(const GenericBoundingBoxTree& A,
                                            const GenericBoundingBoxTree& B,
                                            unsigned int node_A,
                                            unsigned int node_B,
                                            std::vector<unsigned int>& entities_A,
                                            std::vector<unsigned int>& entities_B,
                                            const Mesh* mesh_A,
                                            const Mesh* mesh_B)
{
  // Get bounding boxes for current nodes
  const BBox& bbox_A = A.get_bbox(node_A);
  const BBox& bbox_B = B.get_bbox(node_B);

  // If bounding boxes don't collide, then don't search further
  if (!B.bbox_in_bbox(A.get_bbox_coordinates(node_A), node_B))
    return;

  // Check whether we've reached a leaf in A or B
  const bool is_leaf_A = A.is_leaf(bbox_A, node_A);
  const bool is_leaf_B = B.is_leaf(bbox_B, node_B);

  // If both boxes are leaves (which we know collide), then add them
  if (is_leaf_A && is_leaf_B)
  {
    // child_1 denotes entity for leaves
    const unsigned int entity_index_A = bbox_A.child_1;
    const unsigned int entity_index_B = bbox_B.child_1;

    // If we have a mesh, check that the candidate is really a collision
    if (mesh_A)
    {
      dolfin_assert(mesh_B);
      Cell cell_A(*mesh_A, entity_index_A);
      Cell cell_B(*mesh_B, entity_index_B);
      if (cell_A.collides(cell_B))
      {
        entities_A.push_back(entity_index_A);
        entities_B.push_back(entity_index_B);
      }
    }

    // Otherwise, add the candidate
    else
    {
      entities_A.push_back(entity_index_A);
      entities_B.push_back(entity_index_B);
    }
  }

  // If we reached the leaf in A, then descend B
  else if (is_leaf_A)
  {
    _compute_collisions(A, B, node_A, bbox_B.child_0,
                        entities_A, entities_B, mesh_A, mesh_B);
    _compute_collisions(A, B, node_A, bbox_B.child_1,
                        entities_A, entities_B, mesh_A, mesh_B);
  }

  // If we reached the leaf in B, then descend A
  else if (is_leaf_B)
  {
    _compute_collisions(A, B, bbox_A.child_0, node_B,
                        entities_A, entities_B, mesh_A, mesh_B);
    _compute_collisions(A, B, bbox_A.child_1, node_B,
                        entities_A, entities_B, mesh_A, mesh_B);
  }

  // At this point, we know neither is a leaf so descend the largest
  // tree first. Note that nodes are added in reverse order with the
  // top bounding box at the end so the largest tree (the one with the
  // the most boxes left to traverse) has the largest node number.
  else if (node_A > node_B)
  {
    _compute_collisions(A, B, bbox_A.child_0, node_B,
                        entities_A, entities_B, mesh_A, mesh_B);
    _compute_collisions(A, B, bbox_A.child_1, node_B,
                        entities_A, entities_B, mesh_A, mesh_B);
  }
  else
  {
    _compute_collisions(A, B, node_A, bbox_B.child_0,
                        entities_A, entities_B, mesh_A, mesh_B);
    _compute_collisions(A, B, node_A, bbox_B.child_1,
                        entities_A, entities_B, mesh_A, mesh_B);
  }

  // Note that cases above can be collected in fewer cases but this
  // way the logic is easier to follow.
}
//-----------------------------------------------------------------------------
unsigned int
GenericBoundingBoxTree::_compute_first_collision(const GenericBoundingBoxTree& tree,
                                                 const Point& point,
                                                 unsigned int node)
{
  // Get max integer to signify not found
  unsigned int not_found = std::numeric_limits<unsigned int>::max();

  // Get bounding box for current node
  const BBox& bbox = tree.get_bbox(node);

  // If point is not in bounding box, then don't search further
  if (!tree.point_in_bbox(point.coordinates(), node))
    return not_found;

  // If box is a leaf (which we know contains the point), then return it
  else if (tree.is_leaf(bbox, node))
    return bbox.child_1; // child_1 denotes entity for leaves

  // Check both children
  else
  {
    unsigned int c0 = _compute_first_collision(tree, point, bbox.child_0);
    if (c0 != not_found)
      return c0;

    // Check second child
    unsigned int c1 = _compute_first_collision(tree, point, bbox.child_1);
    if (c1 != not_found)
      return c1;
  }

  // Point not found
  return not_found;
}
//-----------------------------------------------------------------------------
unsigned int
GenericBoundingBoxTree::_compute_first_entity_collision(const GenericBoundingBoxTree& tree,
                                                        const Point& point,
                                                        unsigned int node,
                                                        const Mesh& mesh)
{
  // Get max integer to signify not found
  unsigned int not_found = std::numeric_limits<unsigned int>::max();

  // Get bounding box for current node
  const BBox& bbox = tree.get_bbox(node);

  // If point is not in bounding box, then don't search further
  if (!tree.point_in_bbox(point.coordinates(), node))
    return not_found;

  // If box is a leaf (which we know contains the point), then check entity
  else if (tree.is_leaf(bbox, node))
  {
    // Get entity (child_1 denotes entity index for leaves)
    dolfin_assert(tree._tdim == mesh.topology().dim());
    const unsigned int entity_index = bbox.child_1;
    Cell cell(mesh, entity_index);

    // Check entity
    if (cell.collides(point))
      return entity_index;
  }

  // Check both children
  else
  {
    const unsigned int c0 = _compute_first_entity_collision(tree,
                                                            point,
                                                            bbox.child_0,
                                                            mesh);
    if (c0 != not_found)
      return c0;

    const unsigned int c1 = _compute_first_entity_collision(tree,
                                                            point,
                                                            bbox.child_1,
                                                            mesh);
    if (c1 != not_found)
      return c1;
  }

  // Point not found
  return not_found;
}
//-----------------------------------------------------------------------------
void
GenericBoundingBoxTree::_compute_closest_entity(const GenericBoundingBoxTree& tree,
                                                const Point& point,
                                                unsigned int node,
                                                const Mesh& mesh,
                                                unsigned int& closest_entity,
                                                double& R2)
{
  // Get bounding box for current node
  const BBox& bbox = tree.get_bbox(node);

  // If bounding box is outside radius, then don't search further
  const double r2 = tree.compute_squared_distance_bbox(point.coordinates(), node);
  if (r2 > R2)
    return;

  // If box is leaf (which we know is inside radius), then shrink radius
  else if (tree.is_leaf(bbox, node))
  {
    // Get entity (child_1 denotes entity index for leaves)
    dolfin_assert(tree._tdim == mesh.topology().dim());
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
    _compute_closest_entity(tree, point, bbox.child_0, mesh, closest_entity, R2);
    _compute_closest_entity(tree, point, bbox.child_1, mesh, closest_entity, R2);
  }
}
//-----------------------------------------------------------------------------
void
GenericBoundingBoxTree::_compute_closest_point(const GenericBoundingBoxTree& tree,
                                               const Point& point,
                                               unsigned int node,
                                               unsigned int& closest_point,
                                               double& R2)
{
  // Get bounding box for current node
  const BBox& bbox = tree.get_bbox(node);

  // If box is leaf, then compute distance and shrink radius
  if (tree.is_leaf(bbox, node))
  {
    const double r2 = tree.compute_squared_distance_point(point.coordinates(),
                                                          node);
    if (r2 < R2)
    {
      closest_point = bbox.child_1;
      R2 = r2;
    }
  }
  else
  {
    // If bounding box is outside radius, then don't search further
    const double r2 = tree.compute_squared_distance_bbox(point.coordinates(),
                                                         node);
    if (r2 > R2)
      return;

    // Check both children
    _compute_closest_point(tree, point, bbox.child_0, closest_point, R2);
    _compute_closest_point(tree, point, bbox.child_1, closest_point, R2);
  }
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
  _point_search_tree = create(mesh.geometry().dim());

  // Build tree
  dolfin_assert(_point_search_tree);
  _point_search_tree->build(points);
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
  const unsigned int* vertices = entity.entities(0);
  dolfin_assert(num_vertices >= 2);

  // Get coordinates for first vertex
  const double* x = geometry.x(vertices[0]);
  for (std::size_t j = 0; j < gdim; ++j)
    xmin[j] = xmax[j] = x[j];

  // Compute min and max over remaining vertices
  for (unsigned int i = 1; i < num_vertices; ++i)
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
                                    const std::vector<unsigned int>::iterator& begin,
                                    const std::vector<unsigned int>::iterator& middle,
                                    const std::vector<unsigned int>::iterator& end)
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
std::string GenericBoundingBoxTree::str(bool verbose)
{
  std::stringstream s;
  tree_print(s, _bboxes.size() - 1);
  return s.str();
}
//-----------------------------------------------------------------------------
void GenericBoundingBoxTree::tree_print(std::stringstream& s, unsigned int i)
{
  std::size_t dim = _bbox_coordinates.size() / _bboxes.size();
  std::size_t idx = i*dim;
  s << "[";
  for (unsigned int j = idx; j != idx + dim; ++j)
    s << _bbox_coordinates[j] << " ";
  s << "]\n";

  if (_bboxes[i].child_0 == i)
    s << "leaf containing entity (" << _bboxes[i].child_1 << ")";
  else
  {
    s << "{";
    tree_print(s, _bboxes[i].child_0);
    s << ", \n";
    tree_print(s, _bboxes[i].child_1);
    s << "}\n";
  }
}
//-----------------------------------------------------------------------------
