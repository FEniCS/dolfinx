// Copyright (C) 2019 Matthew Scroggs
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "dofs_permutation.h"
#include "ElementDofLayout.h"
#include <dolfin/common/log.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshIterator.h>

using namespace dolfin;

namespace
{
//-----------------------------------------------------------------------------
int get_num_permutations(mesh::CellType cell_type)
{
  // In general, this will return num_edges + 2*num_faces + 4*num_volumes
  switch (cell_type)
  {
  case (mesh::CellType::point):
    return 0;
  case (mesh::CellType::interval):
    return 1;
  case (mesh::CellType::triangle):
    return 5;
  case (mesh::CellType::tetrahedron):
    return 18;
  case (mesh::CellType::quadrilateral):
    return 6;
  case (mesh::CellType::hexahedron):
    return 28;
  default:
    LOG(WARNING) << "Dof permutations are not defined for this cell type. High "
                    "order elements may be incorrect.";
    return 0;
  }
}
//-----------------------------------------------------------------------------
/// Returns an empty array
/// @param[in] vs The global vertex number of the point's vertex
/// @return An empty array
template <typename T>
Eigen::Array<std::int8_t, 1, Eigen::Dynamic>
    calculate_point_orders(Eigen::Array<T, 1, 1> vs)
{
  return {};
}
//-----------------------------------------------------------------------------
/// Calculates whether or not an interval should be flipped
/// @param[in] vs The global vertex numbers of the interval's vertices
/// @return The reflection order for the interval
template <typename T>
Eigen::Array<std::int8_t, 1, Eigen::Dynamic>
    calculate_interval_orders(Eigen::Array<T, 1, 2> vs)
{
  if (vs[0] > vs[1])
    return Eigen::Array<std::int8_t, 1, 1>(1);
  else
    return Eigen::Array<std::int8_t, 1, 1>(0);
}
//-----------------------------------------------------------------------------
/// Calculates the number of times the rotation and reflection of a triangle
/// should be applied to a triangle with the given global vertex numbers
/// @param[in] vs The global vertex numbers of the triangle's vertices
/// @return The rotation and reflection orders for the triangle
template <typename T>
Eigen::Array<std::int8_t, 1, Eigen::Dynamic>
    calculate_triangle_orders(Eigen::Array<T, 1, 3> vs)
{
  if (vs[0] < vs[1] and vs[0] < vs[2])
    return Eigen::Array<std::int8_t, 1, 2>(0, vs[1] > vs[2]);
  else if (vs[1] < vs[0] and vs[1] < vs[2])
    return Eigen::Array<std::int8_t, 1, 2>(1, vs[2] > vs[0]);
  else if (vs[2] < vs[0] and vs[2] < vs[1])
    return Eigen::Array<std::int8_t, 1, 2>(2, vs[0] > vs[1]);

  throw std::runtime_error("Two of a triangle's vertices appear to be equal.");
}
//-----------------------------------------------------------------------------
/// Calculates the number of times the rotations and reflection of a triangle
/// should be applied to a tetrahedron with the given global vertex numbers
/// @param[in] vs The global vertex numbers of the tetrahedron's vertices
/// @return The rotation and reflection orders for the tetrahedron
template <typename T>
Eigen::Array<std::int8_t, 1, Eigen::Dynamic>
    calculate_tetrahedron_orders(Eigen::Array<T, 1, 4> vs)
{
  if (vs[0] < vs[1] and vs[0] < vs[2] and vs[0] < vs[3])
  {
    const Eigen::Array<std::int8_t, 1, 2> tri_orders
        = calculate_triangle_orders<T>({vs[1], vs[2], vs[3]});
    return Eigen::Array<std::int8_t, 1, 4>(0, 0, tri_orders[0], tri_orders[1]);
  }
  else if (vs[1] < vs[0] and vs[1] < vs[2] and vs[1] < vs[3])
  {
    const Eigen::Array<std::int8_t, 1, 2> tri_orders
        = calculate_triangle_orders<T>({vs[2], vs[0], vs[3]});
    return Eigen::Array<std::int8_t, 1, 4>(1, 0, tri_orders[0], tri_orders[1]);
  }
  else if (vs[2] < vs[0] and vs[2] < vs[1] and vs[2] < vs[3])
  {
    const Eigen::Array<std::int8_t, 1, 2> tri_orders
        = calculate_triangle_orders<T>({vs[0], vs[1], vs[3]});
    return Eigen::Array<std::int8_t, 1, 4>(2, 0, tri_orders[0], tri_orders[1]);
  }
  else if (vs[3] < vs[0] and vs[3] < vs[1] and vs[3] < vs[2])
  {
    const Eigen::Array<std::int8_t, 1, 2> tri_orders
        = calculate_triangle_orders<T>({vs[1], vs[0], vs[2]});
    return Eigen::Array<std::int8_t, 1, 4>(0, 1, tri_orders[0], tri_orders[1]);
  }

  throw std::runtime_error(
      "Two of a tetrahedron's vertices appear to be equal.");
}
//-----------------------------------------------------------------------------
/// Calculates the number of times the rotation and reflection of a
/// quadrilateral should be applied to a quadrilateral with the given global
/// vertex numbers
/// @param[in] v The global vertex numbers of the quadrilateral's vertices
/// @return The rotation and reflection orders for the quadrilateral
template <typename T>
Eigen::Array<std::int8_t, 1, Eigen::Dynamic>
    calculate_quadrilateral_orders(Eigen::Array<T, 1, 4> vs)
{
  if (vs[0] < vs[1] and vs[0] < vs[2] and vs[0] < vs[3])
    return Eigen::Array<std::int8_t, 1, 2>(0, vs[1] > vs[2]);
  else if (vs[1] < vs[0] and vs[1] < vs[2] and vs[1] < vs[3])
    return Eigen::Array<std::int8_t, 1, 2>(1, vs[3] > vs[0]);
  else if (vs[3] < vs[0] and vs[3] < vs[1] and vs[3] < vs[2])
    return Eigen::Array<std::int8_t, 1, 2>(2, vs[2] > vs[1]);
  else if (vs[2] < vs[0] and vs[2] < vs[1] and vs[2] < vs[3])
    return Eigen::Array<std::int8_t, 1, 2>(3, vs[0] > vs[3]);

  throw std::runtime_error(
      "Two of a quadrilateral's vertices appear to be equal.");
}
//-----------------------------------------------------------------------------
/// Calculates the number of times the rotations and reflection of a triangle
/// should be applied to a hexahedron with the given global vertex numbers
/// @param[in] vs The global vertex numbers of the hexahedron's vertices
/// @return The rotation and reflection orders for the hexahedron
template <typename T>
Eigen::Array<std::int8_t, 1, Eigen::Dynamic>
    calculate_hexahedron_orders(Eigen::Array<T, 1, 8> vs)
{
  if (vs[0] < vs[1] and vs[0] < vs[2] and vs[0] < vs[3] and vs[0] < vs[4]
      and vs[0] < vs[5] and vs[0] < vs[6] and vs[0] < vs[7])
  {
    const Eigen::Array<std::int8_t, 1, 2> tri_orders
        = calculate_triangle_orders<T>({vs[1], vs[2], vs[4]});
    return Eigen::Array<std::int8_t, 1, 4>(0, 0, tri_orders[0], tri_orders[1]);
  }
  else if (vs[1] < vs[0] and vs[1] < vs[2] and vs[1] < vs[3] and vs[1] < vs[4]
           and vs[1] < vs[5] and vs[1] < vs[6] and vs[1] < vs[7])
  {
    const Eigen::Array<std::int8_t, 1, 2> tri_orders
        = calculate_triangle_orders<T>({vs[3], vs[0], vs[5]});
    return Eigen::Array<std::int8_t, 1, 4>(1, 0, tri_orders[0], tri_orders[1]);
  }
  else if (vs[2] < vs[0] and vs[2] < vs[1] and vs[2] < vs[3] and vs[2] < vs[4]
           and vs[2] < vs[5] and vs[2] < vs[6] and vs[2] < vs[7])
  {
    const Eigen::Array<std::int8_t, 1, 2> tri_orders
        = calculate_triangle_orders<T>({vs[0], vs[3], vs[6]});
    return Eigen::Array<std::int8_t, 1, 4>(3, 0, tri_orders[0], tri_orders[1]);
  }
  else if (vs[3] < vs[0] and vs[3] < vs[1] and vs[3] < vs[2] and vs[3] < vs[4]
           and vs[3] < vs[5] and vs[3] < vs[6] and vs[3] < vs[7])
  {
    const Eigen::Array<std::int8_t, 1, 2> tri_orders
        = calculate_triangle_orders<T>({vs[1], vs[2], vs[7]});
    return Eigen::Array<std::int8_t, 1, 4>(2, 0, tri_orders[0], tri_orders[1]);
  }
  else if (vs[4] < vs[0] and vs[4] < vs[1] and vs[4] < vs[2] and vs[4] < vs[3]
           and vs[4] < vs[5] and vs[4] < vs[6] and vs[4] < vs[7])
  {
    const Eigen::Array<std::int8_t, 1, 2> tri_orders
        = calculate_triangle_orders<T>({vs[0], vs[6], vs[5]});
    return Eigen::Array<std::int8_t, 1, 4>(0, 1, tri_orders[0], tri_orders[1]);
  }
  else if (vs[5] < vs[0] and vs[5] < vs[1] and vs[5] < vs[2] and vs[5] < vs[3]
           and vs[5] < vs[4] and vs[5] < vs[6] and vs[5] < vs[7])
  {
    const Eigen::Array<std::int8_t, 1, 2> tri_orders
        = calculate_triangle_orders<T>({vs[4], vs[7], vs[1]});
    return Eigen::Array<std::int8_t, 1, 4>(0, 2, tri_orders[0], tri_orders[1]);
  }
  else if (vs[6] < vs[0] and vs[6] < vs[1] and vs[6] < vs[2] and vs[6] < vs[3]
           and vs[6] < vs[4] and vs[6] < vs[5] and vs[6] < vs[7])
  {
    const Eigen::Array<std::int8_t, 1, 2> tri_orders
        = calculate_triangle_orders<T>({vs[7], vs[4], vs[2]});
    return Eigen::Array<std::int8_t, 1, 4>(2, 2, tri_orders[0], tri_orders[1]);
  }
  else if (vs[7] < vs[0] and vs[7] < vs[1] and vs[7] < vs[2] and vs[7] < vs[3]
           and vs[7] < vs[4] and vs[7] < vs[5] and vs[7] < vs[6])
  {
    const Eigen::Array<std::int8_t, 1, 2> tri_orders
        = calculate_triangle_orders<T>({vs[3], vs[5], vs[6]});
    return Eigen::Array<std::int8_t, 1, 4>(2, 1, tri_orders[0], tri_orders[1]);
  }

  throw std::runtime_error(
      "Two of a hexahedron's vertices appear to be equal.");
}
//-----------------------------------------------------------------------------
template <typename T>
std::function<Eigen::Array<std::int8_t, 1, Eigen::Dynamic>(
    Eigen::Array<T, 1, Eigen::Dynamic>)>
get_ordering_function(mesh::CellType cell_type)
{
  switch (cell_type)
  {
  case (mesh::CellType::point):
    return calculate_point_orders<T>;
  case (mesh::CellType::interval):
    return calculate_interval_orders<T>;
  case (mesh::CellType::triangle):
    return calculate_triangle_orders<T>;
  case (mesh::CellType::quadrilateral):
    return calculate_quadrilateral_orders<T>;
  case (mesh::CellType::tetrahedron):
    return calculate_tetrahedron_orders<T>;
  case (mesh::CellType::hexahedron):
    return calculate_hexahedron_orders<T>;
  default:
    throw std::runtime_error("Unrecognised cell type.");
  }
}
//-----------------------------------------------------------------------------
/// Makes a permutation to flip the dofs on an edge
/// @param[in] edge_dofs Number of edge dofs
/// @param[in] blocksize Block size
/// @return The permutation to reverse the dofs
std::vector<int> edge_flip(const int edge_dofs, const int blocksize)
{
  // This will only be called at most once for each mesh
  const int blocks = edge_dofs / blocksize;
  int j = 0;
  std::vector<int> flip(edge_dofs);
  for (int i = 0; i < blocks; ++i)
    for (int k = 0; k < blocksize; ++k)
      flip[j++] = edge_dofs - (i + 1) * blocksize + k;

  return flip;
}
//-----------------------------------------------------------------------------
/// Makes permutations to rotate and reflect the dofs on a triangle
/// @param[in] edge_dofs Number of dofs on the face of the triangle
/// @return Permutations to rotate and reflect the dofs
std::array<std::vector<int>, 2>
triangle_rotation_and_reflection(const int face_dofs, const int blocksize)
{
  // This will only be called at most once for each mesh
  const int blocks = face_dofs / blocksize;
  float root = std::sqrt(8 * blocks + 1);
  assert(root == std::floor(root) && int(root) % 2 == 1);
  int side_length = (root - 1) / 2; // side length of the triangle of face dofs

  std::vector<int> rotation(face_dofs);
  {
    int j = 0;
    int i = 0;
    for (int st = blocks - 1; st >= 0; st -= i)
    {
      ++i;
      int dof = st;
      for (int sub = i + 1; sub <= side_length + 1; ++sub)
      {
        for (int k = 0; k < blocksize; ++k)
          rotation[j++] = blocksize * dof + k;
        dof -= sub;
      }
    }
    assert(j == face_dofs);
  }

  std::vector<int> reflection(face_dofs);
  {
    int j = 0;
    for (int st = 0; st < side_length; ++st)
    {
      int dof = st;
      for (int add = side_length; add > st; --add)
      {
        for (int k = 0; k < blocksize; ++k)
          reflection[j++] = blocksize * dof + k;
        dof += add;
      }
    }
    assert(j == face_dofs);
  }

  return {rotation, reflection};
}
//-----------------------------------------------------------------------------
/// Makes permutations to rotate and reflect the dofs on a quadrilateral
/// @param[in] edge_dofs Number of dofs on the face of the quadrilateral
/// @return Permutations to rotate and reflect the dofs
std::array<std::vector<int>, 2>
quadrilateral_rotation_and_reflection(const int face_dofs, const int blocksize)
{
  // This will only be called at most once for each mesh
  const int blocks = face_dofs / blocksize;
  float root = std::sqrt(blocks);
  assert(root == std::floor(root));
  int side_length = root; // side length of the quadrilateral of face dofs

  std::vector<int> rotation(face_dofs);
  {
    int j = 0;
    for (int st = blocks - side_length; st < blocks; ++st)
      for (int dof = st; dof >= 0; dof -= side_length)
        for (int k = 0; k < blocksize; ++k)
          rotation[j++] = blocksize * dof + k;
    assert(j == face_dofs);
  }

  std::vector<int> reflection(face_dofs);
  {
    int j = 0;
    for (int st = 0; st < side_length; ++st)
      for (int dof = st; dof < blocks; dof += side_length)
        for (int k = 0; k < blocksize; ++k)
          reflection[j++] = blocksize * dof + k;
    assert(j == face_dofs);
  }

  return {rotation, reflection};
}
//-----------------------------------------------------------------------------
/// Makes permutations to rotate and reflect the dofs in a tetrahedron
/// @param[in] volume_dofs Number of dofs on the interior of the
///                        tetrahedron
/// @return Permutations to rotate and reflect the dofs
std::array<std::vector<int>, 4>
tetrahedron_rotations_and_reflection(const int volume_dofs, const int blocksize)
{
  // This will only be called at most once for each mesh
  const int blocks = volume_dofs / blocksize;
  int side_length = 0;
  while (side_length * (side_length + 1) * (side_length + 2) < 6 * blocks)
    ++side_length;
  // side_length is the length of one edge of the tetrahedron of dofs

  assert(side_length * (side_length + 1) * (side_length + 2) == 6 * blocks);

  std::vector<int> rotation1(volume_dofs);
  std::iota(rotation1.begin(), rotation1.end(), 0);
  std::vector<int> rotation2(volume_dofs);
  std::iota(rotation2.begin(), rotation2.end(), 0);

  int start = 0;
  for (int side = side_length; side > 0; --side)
  {
    int face_dofs = blocksize * side * (side + 1) / 2;
    const std::array<std::vector<int>, 2> base_faces
        = triangle_rotation_and_reflection(face_dofs, blocksize);

    std::vector<int> face(face_dofs);
    std::iota(face.begin(), face.end(), start);

    std::vector<int> face2(face_dofs);
    int j = 0;
    int start2 = side * side_length - 1 - side * (side - 1) / 2;
    for (int row = 0; row < side; ++row)
    {
      int dof = start2;
      for (int k = 0; k < blocksize; ++k)
        face2[j++] = dof * blocksize + k;
      for (int sub = 2 + (side_length - side); sub <= side_length - row; ++sub)
      {
        dof -= sub;
        for (int k = 0; k < blocksize; ++k)
          face2[j++] = dof * blocksize + k;
      }
      start2 += (side_length - row - 1) * (side_length - row) / 2;
    }

    start += face_dofs;
    for (std::size_t j = 0; j < face.size(); ++j)
    {
      rotation1[face[j]] = face[base_faces[0][j]];
      rotation2[face2[j]] = face2[base_faces[0][j]];
    }
  }

  std::vector<int> reflection(volume_dofs);
  {
    int j = 0;
    int layerst = 0;
    for (int layer = 0; layer < side_length; ++layer)
    {
      int st = layerst;
      for (int i = side_length; i > layer; --i)
      {
        for (int dof = st; dof < st + i - layer; ++dof)
          for (int k = 0; k < blocksize; ++k)
            reflection[j++] = dof * blocksize + k;
        st += i * (i + 1) / 2 - layer;
      }
      layerst += side_length - layer;
    }
  }

  std::vector<int> rotation3(volume_dofs);
  for (int j = 0; j < volume_dofs; ++j)
    rotation3[j] = rotation2[rotation2[rotation1[j]]];

  return {rotation1, rotation2, rotation3, reflection};
}
//-----------------------------------------------------------------------------
/// Makes permutations to rotate and reflect the dofs in a hexahedron
/// @param[in] volume_dofs Number of dofs on the interior of the
///                        hexahedron
/// @return Permutations to rotate and reflect the dofs
std::array<std::vector<int>, 4>
hexahedron_rotations_and_reflection(const int volume_dofs, const int blocksize)
{
  // This will only be called at most once for each mesh
  const int blocks = volume_dofs / blocksize;
  float root = std::cbrt(blocks);
  assert(root == std::floor(root));
  int side_length = root;
  int area = side_length * side_length;

  std::vector<int> rotation1(volume_dofs);
  {
    int j = 0;
    for (int lst = area - side_length; lst < blocks; lst += area)
      for (int st = lst; st < lst + side_length; ++st)
        for (int dof = st; dof >= lst - area + side_length; dof -= side_length)
          for (int k = 0; k < blocksize; ++k)
            rotation1[j++] = blocksize * dof + k;
    assert(j == volume_dofs);
  }

  std::vector<int> rotation2(volume_dofs);
  {
    int j = 0;
    for (int lst = side_length - 1; lst >= 0; --lst)
      for (int st = lst; st < area; st += side_length)
        for (int dof = st; dof < blocks; dof += area)
          for (int k = 0; k < blocksize; ++k)
            rotation2[j++] = blocksize * dof + k;
    assert(j == volume_dofs);
  }

  std::vector<int> rotation3(volume_dofs);
  {
    int j = 0;
    for (int st = 0; st < area; ++st)
      for (int dof = st; dof < blocks; dof += area)
        for (int k = 0; k < blocksize; ++k)
          rotation3[j++] = blocksize * dof + k;
    assert(j == volume_dofs);
  }

  std::vector<int> reflection(volume_dofs);
  {
    int j = 0;
    for (int lst = 0; lst < area; lst += side_length)
      for (int st = lst; st < blocks; st += area)
        for (int dof = st; dof < st + side_length; ++dof)
          for (int k = 0; k < blocksize; ++k)
            reflection[j++] = blocksize * dof + k;
    assert(j == volume_dofs);
  }

  return {rotation1, rotation2, rotation3, reflection};
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
compute_ordering(const mesh::Mesh& mesh)
{
  mesh::CellType type = mesh.cell_type();
  const int t_dim = mesh.topology().dim();
  const int num_cells = mesh.num_entities(t_dim);
  const int num_permutations = get_num_permutations(type);
  const int num_vertices_per_cell = mesh::num_cell_vertices(type);

  // Find the total number of edges + faces + volumes
  int entity_count = 0;
  for (int dim = 1; dim < t_dim; ++dim)
    entity_count += mesh::cell_num_entities(type, dim);
  entity_count += 1;

  // Get lists of vertices on each entity from the cell type
  std::vector<
      std::pair<std::function<Eigen::Array<std::int8_t, 1, Eigen::Dynamic>(
                    Eigen::Array<std::int32_t, 1, Eigen::Dynamic>)>,
                Eigen::Array<int, 1, Eigen::Dynamic>>>
      entities;
  for (int dim = 1; dim < t_dim; ++dim)
  {
    auto vertices = mesh::get_entity_vertices(type, dim);
    mesh::CellType e_type = mesh::cell_entity_type(type, dim);
    auto f = get_ordering_function<std::int32_t>(
        mesh::cell_entity_type(type, dim));
    // Store the ordering function and vertices associated with the ith
    // entity of dimension dim
    for (int i = 0; i < mesh::cell_num_entities(type, dim); ++i)
      entities.push_back({f, vertices.row(i)});
  }
  { // scope
    // Add on the cell itself as an entity
    Eigen::Array<int, 1, Eigen::Dynamic> row(num_vertices_per_cell);
    auto f = get_ordering_function<std::int32_t>(type);
    for (int i = 0; i < num_vertices_per_cell; ++i)
      row[i] = i;
    entities.push_back({f, row});
  }

  // Set orders for each cell
  Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cell_orders(num_cells, num_permutations);
  const std::vector<std::int64_t>& global_indices
      = mesh.topology().global_indices(0);
  // Reserve memory to store global numbers of vertices of cell
  std::vector<std::int32_t> cell_vs(num_vertices_per_cell);
  for (int cell_n = 0; cell_n < num_cells; ++cell_n)
  {
    // Get the cell and info about it
    const mesh::MeshEntity cell(mesh, t_dim, cell_n);
    const std::int32_t* local_vertices = cell.entities(0);
    for (int i = 0; i < num_vertices_per_cell; ++i)
      cell_vs[i] = global_indices[local_vertices[i]];

    int j = 0;
    // iterate over the cell's entities
    for (std::size_t e_n = 0; e_n < entities.size(); ++e_n)
    {
      // Use the functions for each entites and that entity's global vertex
      // numbers to calculate the number of times each permeutation should be
      // applied on this cell
      auto f = entities[e_n].first;
      Eigen::Array<int, 1, Eigen::Dynamic> v = entities[e_n].second;
      Eigen::Array<int, 1, Eigen::Dynamic> global_v(v.size());
      for (int i = 0; i < v.size(); ++i)
        global_v[i] = cell_vs[v[i]];
      auto orders = f(global_v);
      for (int i = 0; i < orders.size(); ++i)
        cell_orders(cell_n, j++) = orders[i];
    }

    assert(j == num_permutations);
  }
  return cell_orders;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
generate_permutations_interval(const mesh::Mesh& mesh,
                               const fem::ElementDofLayout& dof_layout)
{
  const int num_permutations = get_num_permutations(mesh.cell_type());
  const int dof_count = dof_layout.num_dofs();

  const int edge_dofs = dof_layout.num_entity_dofs(1);
  const int edge_bs = dof_layout.entity_block_size(1);

  int perm_n = 0;

  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      permutations(num_permutations, dof_count);
  for (int i = 0; i < permutations.cols(); ++i)
    permutations.col(i) = i;

  // Make edge flipping permutations
  const std::vector<int> base_flip = edge_flip(edge_dofs, edge_bs);
  const Eigen::Array<int, Eigen::Dynamic, 1> edge
      = dof_layout.entity_dofs(1, 0);
  for (std::size_t i = 0; i < base_flip.size(); ++i)
    permutations(perm_n, edge(i)) = edge(base_flip[i]);
  ++perm_n;

  assert(perm_n == get_num_permutations(mesh::CellType::interval));

  return permutations;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
generate_permutations_triangle(const mesh::Mesh& mesh,
                               const fem::ElementDofLayout& dof_layout)
{
  const int num_permutations = get_num_permutations(mesh.cell_type());
  const int dof_count = dof_layout.num_dofs();

  const int edge_dofs = dof_layout.num_entity_dofs(1);
  const int face_dofs = dof_layout.num_entity_dofs(2);
  const int edge_bs = dof_layout.entity_block_size(1);
  const int face_bs = dof_layout.entity_block_size(2);

  int perm_n = 0;

  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      permutations(num_permutations, dof_count);
  for (int i = 0; i < permutations.cols(); ++i)
    permutations.col(i) = i;

  // Make edge flipping permutations
  const std::vector<int> base_flip = edge_flip(edge_dofs, edge_bs);
  for (int edge_n = 0; edge_n < 3; ++edge_n)
  {
    const Eigen::Array<int, Eigen::Dynamic, 1> edge
        = dof_layout.entity_dofs(1, edge_n);
    for (std::size_t i = 0; i < base_flip.size(); ++i)
      permutations(perm_n, edge(i)) = edge(base_flip[i]);
    ++perm_n;
  }
  // Make permutations that rotate and reflect the face dofs
  const std::array<std::vector<int>, 2> base_faces
      = triangle_rotation_and_reflection(face_dofs, face_bs);
  const Eigen::Array<int, Eigen::Dynamic, 1> face
      = dof_layout.entity_dofs(2, 0);
  for (int f_n = 0; f_n < 2; ++f_n)
  {
    for (std::size_t i = 0; i < base_faces[f_n].size(); ++i)
      permutations(perm_n, face(i)) = face(base_faces[f_n][i]);
    ++perm_n;
  }

  assert(perm_n == get_num_permutations(mesh::CellType::triangle));

  return permutations;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
generate_permutations_quadrilateral(const mesh::Mesh& mesh,
                                    const fem::ElementDofLayout& dof_layout)
{
  const int num_permutations = get_num_permutations(mesh.cell_type());
  const int dof_count = dof_layout.num_dofs();

  const int edge_dofs = dof_layout.num_entity_dofs(1);
  const int face_dofs = dof_layout.num_entity_dofs(2);
  const int edge_bs = dof_layout.entity_block_size(1);
  const int face_bs = dof_layout.entity_block_size(2);

  int perm_n = 0;

  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      permutations(num_permutations, dof_count);
  for (int i = 0; i < permutations.cols(); ++i)
    permutations.col(i) = i;

  // Make edge flipping permutations
  const std::vector<int> base_flip = edge_flip(edge_dofs, edge_bs);
  for (int edge_n = 0; edge_n < 4; ++edge_n)
  {
    const Eigen::Array<int, Eigen::Dynamic, 1> edge
        = dof_layout.entity_dofs(1, edge_n);
    for (std::size_t i = 0; i < base_flip.size(); ++i)
      permutations(perm_n, edge(i)) = edge(base_flip[i]);
    ++perm_n;
  }
  // Make permutations that rotate and reflect the face dofs
  const std::array<std::vector<int>, 2> base_faces
      = quadrilateral_rotation_and_reflection(face_dofs, face_bs);
  const Eigen::Array<int, Eigen::Dynamic, 1> face
      = dof_layout.entity_dofs(2, 0);
  for (int f_n = 0; f_n < 2; ++f_n)
  {
    for (std::size_t i = 0; i < base_faces[f_n].size(); ++i)
      permutations(perm_n, face(i)) = face(base_faces[f_n][i]);
    ++perm_n;
  }

  assert(perm_n == get_num_permutations(mesh::CellType::quadrilateral));

  return permutations;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
generate_permutations_tetrahedron(const mesh::Mesh& mesh,
                                  const fem::ElementDofLayout& dof_layout)
{
  const int num_permutations = get_num_permutations(mesh.cell_type());
  const int dof_count = dof_layout.num_dofs();

  const int edge_dofs = dof_layout.num_entity_dofs(1);
  const int face_dofs = dof_layout.num_entity_dofs(2);
  const int volume_dofs = dof_layout.num_entity_dofs(3);
  const int edge_bs = dof_layout.entity_block_size(1);
  const int face_bs = dof_layout.entity_block_size(2);
  const int volume_bs = dof_layout.entity_block_size(3);

  int perm_n = 0;

  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      permutations(num_permutations, dof_count);
  for (int i = 0; i < permutations.cols(); ++i)
    permutations.col(i) = i;

  // Make edge flipping permutations
  std::vector<int> base_flip = edge_flip(edge_dofs, edge_bs);
  for (int edge_n = 0; edge_n < 6; ++edge_n)
  {
    const Eigen::Array<int, Eigen::Dynamic, 1> edge
        = dof_layout.entity_dofs(1, edge_n);
    for (std::size_t i = 0; i < base_flip.size(); ++i)
      permutations(perm_n, edge(i)) = edge(base_flip[i]);
    ++perm_n;
  }
  // Make permutations that rotate and reflect the face dofs
  const std::array<std::vector<int>, 2> base_faces
      = triangle_rotation_and_reflection(face_dofs, face_bs);
  for (int face_n = 0; face_n < 4; ++face_n)
  {
    const Eigen::Array<int, Eigen::Dynamic, 1> face
        = dof_layout.entity_dofs(2, face_n);
    for (int f_n = 0; f_n < 2; ++f_n)
    {
      for (std::size_t i = 0; i < base_faces[f_n].size(); ++i)
        permutations(perm_n, face(i)) = face(base_faces[f_n][i]);
      ++perm_n;
    }
  }
  // Make permutations that rotate and reflect the volume dofs
  const std::array<std::vector<int>, 4> base_volumes
      = tetrahedron_rotations_and_reflection(volume_dofs, volume_bs);
  const Eigen::Array<int, Eigen::Dynamic, 1> volume
      = dof_layout.entity_dofs(3, 0);
  for (int v_n = 0; v_n < 4; ++v_n)
  {
    for (std::size_t i = 0; i < base_volumes[v_n].size(); ++i)
      permutations(perm_n, volume(i)) = volume(base_volumes[v_n][i]);
    ++perm_n;
  }

  assert(perm_n == get_num_permutations(mesh::CellType::tetrahedron));

  return permutations;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
generate_permutations_hexahedron(const mesh::Mesh& mesh,
                                 const fem::ElementDofLayout& dof_layout)
{
  const int num_permutations = get_num_permutations(mesh.cell_type());
  const int dof_count = dof_layout.num_dofs();

  const int edge_dofs = dof_layout.num_entity_dofs(1);
  const int face_dofs = dof_layout.num_entity_dofs(2);
  const int volume_dofs = dof_layout.num_entity_dofs(3);
  const int edge_bs = dof_layout.entity_block_size(1);
  const int face_bs = dof_layout.entity_block_size(2);
  const int volume_bs = dof_layout.entity_block_size(3);

  int perm_n = 0;

  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      permutations(num_permutations, dof_count);
  for (int i = 0; i < permutations.cols(); ++i)
    permutations.col(i) = i;

  // Make edge flipping permutations
  std::vector<int> base_flip = edge_flip(edge_dofs, edge_bs);
  for (int edge_n = 0; edge_n < 12; ++edge_n)
  {
    const Eigen::Array<int, Eigen::Dynamic, 1> edge
        = dof_layout.entity_dofs(1, edge_n);
    for (std::size_t i = 0; i < base_flip.size(); ++i)
      permutations(perm_n, edge(i)) = edge(base_flip[i]);
    ++perm_n;
  }
  // Make permutations that rotate and reflect the face dofs
  const std::array<std::vector<int>, 2> base_faces
      = quadrilateral_rotation_and_reflection(face_dofs, face_bs);
  for (int face_n = 0; face_n < 6; ++face_n)
  {
    const Eigen::Array<int, Eigen::Dynamic, 1> face
        = dof_layout.entity_dofs(2, face_n);
    for (int f_n = 0; f_n < 2; ++f_n)
    {
      for (std::size_t i = 0; i < base_faces[f_n].size(); ++i)
        permutations(perm_n, face(i)) = face(base_faces[f_n][i]);
      ++perm_n;
    }
  }
  // Make permutations that rotate and reflect the volume dofs
  const std::array<std::vector<int>, 4> base_volumes
      = hexahedron_rotations_and_reflection(volume_dofs, volume_bs);
  const Eigen::Array<int, Eigen::Dynamic, 1> volume
      = dof_layout.entity_dofs(3, 0);
  for (int v_n = 0; v_n < 4; ++v_n)
  {
    for (std::size_t i = 0; i < base_volumes[v_n].size(); ++i)
      permutations(perm_n, volume(i)) = volume(base_volumes[v_n][i]);
    ++perm_n;
  }

  assert(perm_n == get_num_permutations(mesh::CellType::hexahedron));

  return permutations;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
generate_permutations(const mesh::Mesh& mesh,
                      const fem::ElementDofLayout& dof_layout)
{
  // FIXME: What should be done here if the dof layout shape does not match the
  // shape of the element (eg AAF on the periodical table of finite elements)
  if (dof_layout.num_sub_dofmaps() == 0)
  {
    switch (mesh.cell_type())
    {
    case (mesh::CellType::point):
      // For a point, _permutations will have 0 rows, and _cell_ordering will
      // have 0 columns, so the for loops that apply the permutations will be
      // empty
      break;
    case (mesh::CellType::interval):
      return generate_permutations_interval(mesh, dof_layout);
    case (mesh::CellType::triangle):
      return generate_permutations_triangle(mesh, dof_layout);
    case (mesh::CellType::tetrahedron):
      return generate_permutations_tetrahedron(mesh, dof_layout);
    case (mesh::CellType::quadrilateral):
      return generate_permutations_quadrilateral(mesh, dof_layout);
    case (mesh::CellType::hexahedron):
      return generate_permutations_hexahedron(mesh, dof_layout);

    default:
      // The switch should exit before this is reached
      throw std::runtime_error("Unrecognised cell type.");
    }
  }

  // If there are subdofmaps
  const int num_permutations = get_num_permutations(mesh.cell_type());
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> output(
      num_permutations, dof_layout.num_dofs());
  for (int i = 0; i < dof_layout.num_sub_dofmaps(); ++i)
  {
    const std::vector<int> sub_view = dof_layout.sub_view({i});
    const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        sub_perm = generate_permutations(mesh, *dof_layout.sub_dofmap({i}));
    assert(unsigned(sub_view.size()) == sub_perm.cols());
    for (int p = 0; p < num_permutations; ++p)
      for (std::size_t j = 0; j < sub_view.size(); ++j)
        output(p, sub_view[j]) = sub_view[sub_perm(p, j)];
  }
  return output;
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
fem::compute_dof_permutations(const mesh::Mesh& mesh,
                              const fem::ElementDofLayout& dof_layout)
{
  // Build ordering in each cell. It stores the number of times each row
  // of _permutations should be applied on each cell Will have shape
  // (number of cells) × (number of permutations)
  Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cell_ordering = compute_ordering(mesh);

  // Build permutations. Each row of this represent the rotation or
  // reflection of a mesh entity Will have shape (number of
  // permutations) × (number of dofs on reference) where (number of
  // permutations) = (num_edges + 2*num_faces + 4*num_volumes)
  const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      permutations = generate_permutations(mesh, dof_layout);

  // Compute permutations on each cell
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p(
      cell_ordering.rows(), permutations.cols());
  for (int i = 0; i < p.cols(); ++i)
    p.col(i) = i;

  // For each cell
  std::vector<int> temp(p.cols());
  for (int cell = 0; cell < cell_ordering.rows(); ++cell)
  {
    // For each permutation in permutations
    for (int i = 0; i < cell_ordering.cols(); ++i)
    {
      // cell_ordering(cell, i) says how many times this permutation
      // should be applied
      for (int j = 0; j < cell_ordering(cell, i); ++j)
      {
        // This must be inside the loop as p changes after each
        // permutation
        for (int k = 0; k < p.cols(); ++k)
          temp[k] = p(cell, k);
        for (int k = 0; k < p.cols(); ++k)
          p(cell, permutations(i, k)) = temp[k];
      }
    }
  }

  return p;
}
//-----------------------------------------------------------------------------
