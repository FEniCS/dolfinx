// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "BoundingBoxTree.h"
#include "CollisionPredicates.h"
#include <dolfin/mesh/Geometry.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>

using namespace dolfin;

namespace
{
//-----------------------------------------------------------------------------
// Check whether point is outside region defined by facet ABC. The
// fourth vertex is needed to define the orientation.
bool point_outside_of_plane(const Eigen::Vector3d& point,
                            const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                            const Eigen::Vector3d& c, const Eigen::Vector3d& d)
{
  // Algorithm from Real-time collision detection by Christer Ericson:
  // PointOutsideOfPlane on page 144, Section 5.1.6.
  const Eigen::Vector3d v = (b - a).cross(c - a);
  const double signp = v.dot(point - a);
  const double signd = v.dot(d - a);
  return signp * signd < 0.0;
}
//-----------------------------------------------------------------------------
// Check whether bounding box is a leaf node
bool is_leaf(const geometry::BoundingBoxTree::BBox& bbox, int node)
{
  // Leaf nodes are marked by setting child_0 equal to the node itself
  return bbox[0] == node;
}
//-----------------------------------------------------------------------------
// Compute collisions with point (recursive)
void _compute_collisions_point(const geometry::BoundingBoxTree& tree,
                               const Eigen::Vector3d& point, int node,
                               const mesh::Mesh* mesh,
                               std::vector<int>& entities)
{
  // Get bounding box for current node
  const geometry::BoundingBoxTree::BBox bbox = tree.bbox(node);

  if (!tree.point_in_bbox(point, node))
  {
    // If point is not in bounding box, then don't search further
    return;
  }
  else if (is_leaf(bbox, node))
  {
    // If box is a leaf (which we know contains the point), then add it

    // child_1 denotes entity for leaves
    const int entity_index = bbox[1];

    // If we have a mesh, check that the candidate is really a collision
    if (mesh)
    {
      // Get cell
      mesh::MeshEntity cell(*mesh, mesh->topology().dim(), entity_index);
      if (geometry::CollisionPredicates::collides(cell, point))
        entities.push_back(entity_index);
    }

    // Otherwise, add the candidate
    else
      entities.push_back(entity_index);
  }
  else
  {
    // Check both children
    _compute_collisions_point(tree, point, bbox[0], mesh, entities);
    _compute_collisions_point(tree, point, bbox[1], mesh, entities);
  }
}
//-----------------------------------------------------------------------------
// Compute first collision (recursive)
int _compute_first_collision(const geometry::BoundingBoxTree& tree,
                             const Eigen::Vector3d& point, int node)
{
  // Get bounding box for current node
  const geometry::BoundingBoxTree::BBox bbox = tree.bbox(node);

  if (!tree.point_in_bbox(point, node))
  {
    // If point is not in bounding box, then don't search further
    return -1;
  }
  else if (is_leaf(bbox, node))
  {
    // If box is a leaf (which we know contains the point), then return it
    return bbox[1]; // child_1 denotes entity for leaves
  }
  else
  {
    // Check both children
    int c0 = _compute_first_collision(tree, point, bbox[0]);
    if (c0 >= 0)
      return c0;

    // Check second child
    int c1 = _compute_first_collision(tree, point, bbox[1]);
    if (c1 >= 0)
      return c1;
  }

  // Point not found
  return -1;
}
//-----------------------------------------------------------------------------
// Compute first entity collision (recursive)
int _compute_first_entity_collision(const geometry::BoundingBoxTree& tree,
                                    const Eigen::Vector3d& point, int node,
                                    const mesh::Mesh& mesh)
{
  // Get bounding box for current node
  const geometry::BoundingBoxTree::BBox bbox = tree.bbox(node);

  // If point is not in bounding box, then don't search further
  if (!tree.point_in_bbox(point, node))
  {
    // If point is not in bounding box, then don't search further
    return -1;
  }
  else if (is_leaf(bbox, node))
  {
    // If box is a leaf (which we know contains the point), then check entity

    // Get entity (child_1 denotes entity index for leaves)
    assert(tree.tdim == mesh.topology().dim());
    const int entity_index = bbox[1];
    mesh::MeshEntity cell(mesh, mesh.topology().dim(), entity_index);

    // Check entity
    if (geometry::CollisionPredicates::collides(cell, point))
      return entity_index;
  }
  else
  {
    // Check both children
    const int c0 = _compute_first_entity_collision(tree, point, bbox[0], mesh);
    if (c0 >= 0)
      return c0;

    const int c1 = _compute_first_entity_collision(tree, point, bbox[1], mesh);
    if (c1 >= 0)
      return c1;
  }

  // Point not found
  return -1;
}
//-----------------------------------------------------------------------------
// Compute collisions with tree (recursive)
void _compute_collisions_tree(const geometry::BoundingBoxTree& A,
                              const geometry::BoundingBoxTree& B, int node_A,
                              int node_B, const mesh::Mesh* mesh_A,
                              const mesh::Mesh* mesh_B,
                              std::vector<int>& entities_A,
                              std::vector<int>& entities_B)
{
  // Get bounding boxes for current nodes
  const geometry::BoundingBoxTree::BBox bbox_A = A.bbox(node_A);
  const geometry::BoundingBoxTree::BBox bbox_B = B.bbox(node_B);

  // If bounding boxes don't collide, then don't search further
  if (!B.bbox_in_bbox(A.get_bbox_coordinates(node_A), node_B))
    return;

  // Check whether we've reached a leaf in A or B
  const bool is_leaf_A = is_leaf(bbox_A, node_A);
  const bool is_leaf_B = is_leaf(bbox_B, node_B);
  if (is_leaf_A and is_leaf_B)
  {
    // If both boxes are leaves (which we know collide), then add them

    // child_1 denotes entity for leaves
    const int entity_index_A = bbox_A[1];
    const int entity_index_B = bbox_B[1];

    // If we have a mesh, check that the candidate is really a collision
    if (mesh_A)
    {
      assert(mesh_B);
      mesh::MeshEntity cell_A(*mesh_A, mesh_A->topology().dim(),
                              entity_index_A);
      mesh::MeshEntity cell_B(*mesh_B, mesh_B->topology().dim(),
                              entity_index_B);
      if (geometry::CollisionPredicates::collides(cell_A, cell_B))
      {
        entities_A.push_back(entity_index_A);
        entities_B.push_back(entity_index_B);
      }
    }
    else
    {
      // Otherwise, add the candidate
      entities_A.push_back(entity_index_A);
      entities_B.push_back(entity_index_B);
    }
  }
  else if (is_leaf_A)
  {
    // If we reached the leaf in A, then descend B
    _compute_collisions_tree(A, B, node_A, bbox_B[0], mesh_A, mesh_B,
                             entities_A, entities_B);
    _compute_collisions_tree(A, B, node_A, bbox_B[1], mesh_A, mesh_B,
                             entities_A, entities_B);
  }
  else if (is_leaf_B)
  {
    // If we reached the leaf in B, then descend A
    _compute_collisions_tree(A, B, bbox_A[0], node_B, mesh_A, mesh_B,
                             entities_A, entities_B);
    _compute_collisions_tree(A, B, bbox_A[1], node_B, mesh_A, mesh_B,
                             entities_A, entities_B);
  }
  else if (node_A > node_B)
  {
    // At this point, we know neither is a leaf so descend the largest
    // tree first. Note that nodes are added in reverse order with the top
    // bounding box at the end so the largest tree (the one with the the
    // most boxes left to traverse) has the largest node number.
    _compute_collisions_tree(A, B, bbox_A[0], node_B, mesh_A, mesh_B,
                             entities_A, entities_B);
    _compute_collisions_tree(A, B, bbox_A[1], node_B, mesh_A, mesh_B,
                             entities_A, entities_B);
  }
  else
  {
    _compute_collisions_tree(A, B, node_A, bbox_B[0], mesh_A, mesh_B,
                             entities_A, entities_B);
    _compute_collisions_tree(A, B, node_A, bbox_B[1], mesh_A, mesh_B,
                             entities_A, entities_B);
  }

  // Note that cases above can be collected in fewer cases but this way
  // the logic is easier to follow.
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
std::pair<std::vector<int>, std::vector<int>>
geometry::compute_collisions(const BoundingBoxTree& tree0,
                             const BoundingBoxTree& tree1)
{
  // Create data structures for storing collisions
  std::vector<int> entities_0;
  std::vector<int> entities_1;

  // Call recursive find function
  _compute_collisions_tree(tree0, tree1, tree0.num_bboxes() - 1,
                           tree1.num_bboxes() - 1, nullptr, nullptr, entities_0,
                           entities_1);

  return std::make_pair(entities_0, entities_1);
}
//-----------------------------------------------------------------------------
std::pair<std::vector<int>, std::vector<int>>
geometry::compute_entity_collisions(const BoundingBoxTree& tree0,
                                    const BoundingBoxTree& tree1,
                                    const mesh::Mesh& mesh0,
                                    const mesh::Mesh& mesh1)
{
  // Create data structures for storing collisions
  std::vector<int> entities_0;
  std::vector<int> entities_1;

  // Call recursive find function
  _compute_collisions_tree(tree0, tree1, tree0.num_bboxes() - 1,
                           tree1.num_bboxes() - 1, &mesh0, &mesh1, entities_0,
                           entities_1);

  return std::make_pair(entities_0, entities_1);
}
//-----------------------------------------------------------------------------
std::vector<int> geometry::compute_collisions(const BoundingBoxTree& tree,
                                              const Eigen::Vector3d& point)
{
  // Call recursive find function
  std::vector<int> entities;
  _compute_collisions_point(tree, point, tree.num_bboxes() - 1, nullptr,
                            entities);
  return entities;
}
//-----------------------------------------------------------------------------
std::vector<int>
geometry::compute_entity_collisions(const BoundingBoxTree& tree,
                                    const Eigen::Vector3d& point,
                                    const mesh::Mesh& mesh)
{
  // Point in entity only implemented for cells. Consider extending this.
  if (tree.tdim != mesh.topology().dim())
  {
    throw std::runtime_error(
        "Cannot compute collision between point and mesh entities. "
        "Point-in-entity is only implemented for cells");
  }

  // Call recursive find function to compute bounding box candidates
  std::vector<int> entities;
  _compute_collisions_point(tree, point, tree.num_bboxes() - 1, &mesh,
                            entities);

  return entities;
}
//-----------------------------------------------------------------------------
int geometry::compute_first_collision(const BoundingBoxTree& tree,
                                      const Eigen::Vector3d& point)
{
  // Call recursive find function
  return _compute_first_collision(tree, point, tree.num_bboxes() - 1);
}
//-----------------------------------------------------------------------------
int geometry::compute_first_entity_collision(const BoundingBoxTree& tree,
                                             const Eigen::Vector3d& point,
                                             const mesh::Mesh& mesh)
{
  // Point in entity only implemented for cells. Consider extending this.
  if (tree.tdim != mesh.topology().dim())
  {
    throw std::runtime_error(
        "Cannot compute collision between point and mesh entities. "
        "Point-in-entity is only implemented for cells");
  }

  // Call recursive find function
  return _compute_first_entity_collision(tree, point, tree.num_bboxes() - 1,
                                         mesh);
}
//-----------------------------------------------------------------------------
bool geometry::collides(const geometry::BoundingBoxTree& tree,
                        const Eigen::Vector3d& point)
{
  return geometry::compute_first_collision(tree, point) >= 0;
}
//-----------------------------------------------------------------------------
bool geometry::collides_entity(const geometry::BoundingBoxTree& tree,
                               const Eigen::Vector3d& point,
                               const mesh::Mesh& mesh)
{
  return geometry::compute_first_entity_collision(tree, point, mesh) >= 0;
}
//-----------------------------------------------------------------------------
std::vector<int>
geometry::compute_process_collisions(const geometry::BoundingBoxTree& tree,
                                     const Eigen::Vector3d& point)
{
  if (tree.global_tree)
    return geometry::compute_collisions(*tree.global_tree, point);
  else
  {
    std::vector<int> collision;
    if (tree.point_in_bbox(point, tree.num_bboxes() - 1))
      collision.push_back(0);
    return collision;
  }
}
//-----------------------------------------------------------------------------
double geometry::squared_distance(const mesh::MeshEntity& entity,
                                  const Eigen::Vector3d& point)
{
  const mesh::CellType type = entity.mesh().cell_type;
  const mesh::Geometry& geometry = entity.mesh().geometry();
  switch (type)
  {
  case (mesh::CellType::interval):
  {
    const std::int32_t* vertices = entity.entities(0);
    const Eigen::Vector3d a = geometry.x(vertices[0]);
    const Eigen::Vector3d b = geometry.x(vertices[1]);
    return geometry::squared_distance_interval(point, a, b);
  }
  case (mesh::CellType::triangle):
  {
    const std::int32_t* vertices = entity.entities(0);
    const Eigen::Vector3d a = geometry.x(vertices[0]);
    const Eigen::Vector3d b = geometry.x(vertices[1]);
    const Eigen::Vector3d c = geometry.x(vertices[2]);
    return geometry::squared_distance_triangle(point, a, b, c);
  }
  case (mesh::CellType::tetrahedron):
  {
    // Algorithm from Real-time collision detection by Christer Ericson:
    // ClosestPtPointTetrahedron on page 143, Section 5.1.6.
    //
    // Note: This algorithm actually computes the closest point but we
    // only return the distance to that point.

    // Get the vertices as points
    const std::int32_t* vertices = entity.entities(0);
    const Eigen::Vector3d a = geometry.x(vertices[0]);
    const Eigen::Vector3d b = geometry.x(vertices[1]);
    const Eigen::Vector3d c = geometry.x(vertices[2]);
    const Eigen::Vector3d d = geometry.x(vertices[3]);

    // Initialize squared distance
    double r2 = std::numeric_limits<double>::max();

    // Check face ABC
    if (point_outside_of_plane(point, a, b, c, d))
      r2 = std::min(r2, geometry::squared_distance_triangle(point, a, b, c));

    // Check face ACD
    if (point_outside_of_plane(point, a, c, d, b))
      r2 = std::min(r2, geometry::squared_distance_triangle(point, a, c, d));

    // Check face ADB
    if (point_outside_of_plane(point, a, d, b, c))
      r2 = std::min(r2, geometry::squared_distance_triangle(point, a, d, b));

    // Check facet BDC
    if (point_outside_of_plane(point, b, d, c, a))
      r2 = std::min(r2, geometry::squared_distance_triangle(point, b, d, c));

    // Point is inside tetrahedron so distance is zero
    if (r2 == std::numeric_limits<double>::max())
      r2 = 0.0;

    return r2;
  }
  default:
    throw std::invalid_argument(
        "cell_normal not supported for this cell type.");
  }
  return 0.0;
}
//-----------------------------------------------------------------------------
double geometry::squared_distance_triangle(const Eigen::Vector3d& point,
                                           const Eigen::Vector3d& a,
                                           const Eigen::Vector3d& b,
                                           const Eigen::Vector3d& c)
{
  // Algorithm from Real-time collision detection by Christer Ericson:
  // Closest Pt Point Triangle on page 141, Section 5.1.5.
  //
  // Algorithm modified to handle triangles embedded in 3D.
  //
  // Note: This algorithm actually computes the closest point but we
  // only return the distance to that point.

  // Compute normal to plane defined by triangle
  const Eigen::Vector3d ab = b - a;
  const Eigen::Vector3d ac = c - a;
  Eigen::Vector3d n = ab.cross(ac);
  n /= n.norm();

  // Subtract projection onto plane
  const double pn = (point - a).dot(n);
  const Eigen::Vector3d p = point - n * pn;

  // Check if point is in vertex region outside A
  const Eigen::Vector3d ap = p - a;
  const double d1 = ab.dot(ap);
  const double d2 = ac.dot(ap);
  if (d1 <= 0.0 && d2 <= 0.0)
    return ap.squaredNorm() + pn * pn;

  // Check if point is in vertex region outside B
  const Eigen::Vector3d bp = p - b;
  const double d3 = ab.dot(bp);
  const double d4 = ac.dot(bp);
  if (d3 >= 0.0 && d4 <= d3)
    return bp.squaredNorm() + pn * pn;

  // Check if point is in edge region of AB and if so compute projection
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
  {
    const double v = d1 / (d1 - d3);
    return (a + ab * v - p).squaredNorm() + pn * pn;
  }

  // Check if point is in vertex region outside C
  const Eigen::Vector3d cp = p - c;
  const double d5 = ab.dot(cp);
  const double d6 = ac.dot(cp);
  if (d6 >= 0.0 && d5 <= d6)
    return cp.squaredNorm() + pn * pn;

  // Check if point is in edge region of AC and if so compute projection
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
  {
    const double w = d2 / (d2 - d6);
    return (a + ac * w - p).squaredNorm() + pn * pn;
  }

  // Check if point is in edge region of BC and if so compute projection
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
  {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return (b + (c - b) * w - p).squaredNorm() + pn * pn;
  }

  // Point is inside triangle so return distance to plane
  return pn * pn;
}
//-----------------------------------------------------------------------------
double geometry::squared_distance_interval(const Eigen::Vector3d& point,
                                           const Eigen::Vector3d& a,
                                           const Eigen::Vector3d& b)
{
  // Compute vector
  const Eigen::Vector3d v0 = point - a;
  const Eigen::Vector3d v1 = point - b;
  const Eigen::Vector3d v01 = b - a;

  // Check if a is closest point (outside of interval)
  const double a0 = v0.dot(v01);
  if (a0 < 0.0)
    return v0.dot(v0);

  // Check if b is closest point (outside the interval)
  const double a1 = -v1.dot(v01);
  if (a1 < 0.0)
    return v1.dot(v1);

  // Inside interval, so use Pythagoras to subtract length of projection
  return std::max(v0.dot(v0) - a0 * a0 / v01.dot(v01), 0.0);
}
//-----------------------------------------------------------------------------
