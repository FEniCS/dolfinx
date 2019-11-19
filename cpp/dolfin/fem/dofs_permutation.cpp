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
/// Calculates the number of times the rotation and reflection of a triangle
/// should be applied to a triangle with the given global vertex numbers
/// @param[in] v1 The global vertex number of the triangle's first vertex
/// @param[in] v2 The global vertex number of the triangle's second vertex
/// @param[in] v3 The global vertex number of the triangle's third vertex
/// @return The rotation and reflection orders for the triangle
std::array<int, 2> calculate_triangle_orders(int v1, int v2, int v3)
{
  if (v1 < v2 and v1 < v3)
    return {0, v2 > v3};
  else if (v2 < v1 and v2 < v3)
    return {1, v3 > v1};
  else if (v3 < v1 and v3 < v2)
    return {2, v1 > v2};

  throw std::runtime_error("Two of a triangle's vertices appear to be equal.");
}
//-----------------------------------------------------------------------------
/// Calculates the number of times the rotations and reflection of a triangle
/// should be applied to a tetrahedron with the given global vertex numbers
/// @param[in] v1 The global vertex number of the tetrahedron's first vertex
/// @param[in] v2 The global vertex number of the tetrahedron's second vertex
/// @param[in] v3 The global vertex number of the tetrahedron's third vertex
/// @param[in] v4 The global vertex number of the tetrahedron's fourth vertex
/// @return The rotation and reflection orders for the tetrahedron
std::array<int, 4> calculate_tetrahedron_orders(int v1, int v2, int v3, int v4)
{
  if (v1 < v2 and v1 < v3 and v1 < v4)
  {
    const std::array<int, 2> tri_orders = calculate_triangle_orders(v2, v3, v4);
    return {0, 0, tri_orders[0], tri_orders[1]};
  }
  else if (v2 < v1 and v2 < v3 and v2 < v4)
  {
    const std::array<int, 2> tri_orders = calculate_triangle_orders(v3, v1, v4);
    return {1, 0, tri_orders[0], tri_orders[1]};
  }
  else if (v3 < v1 and v3 < v2 and v3 < v4)
  {
    const std::array<int, 2> tri_orders = calculate_triangle_orders(v1, v2, v4);
    return {2, 0, tri_orders[0], tri_orders[1]};
  }
  else if (v4 < v1 and v4 < v2 and v4 < v3)
  {
    const std::array<int, 2> tri_orders = calculate_triangle_orders(v2, v1, v3);
    return {0, 1, tri_orders[0], tri_orders[1]};
  }

  throw std::runtime_error(
      "Two of a tetrahedron's vertices appear to be equal.");
}
//-----------------------------------------------------------------------------
/// Calculates the number of times the rotation and reflection of a
/// quadrilateral should be applied to a quadrilateral with the given global
/// vertex numbers
/// @param[in] v1 The global vertex number of the quadrilateral's first vertex
/// @param[in] v2 The global vertex number of the quadrilateral's second vertex
/// @param[in] v3 The global vertex number of the quadrilateral's third vertex
/// @param[in] v4 The global vertex number of the quadrilateral's fourth vertex
/// @return The rotation and reflection orders for the quadrilateral
std::array<int, 2> calculate_quadrilateral_orders(int v1, int v2, int v3,
                                                  int v4)
{
  if (v1 < v2 and v1 < v3 and v1 < v4)
    return {0, v2 > v3};
  else if (v2 < v1 and v2 < v3 and v2 < v4)
    return {1, v4 > v1};
  else if (v4 < v1 and v4 < v2 and v4 < v3)
    return {2, v3 > v2};
  else if (v3 < v1 and v3 < v2 and v3 < v4)
    return {3, v1 > v4};

  throw std::runtime_error(
      "Two of a quadrilateral's vertices appear to be equal.");
}
//-----------------------------------------------------------------------------
/// Calculates the number of times the rotations and reflection of a triangle
/// should be applied to a hexahedron with the given global vertex numbers
/// @param[in] v1 The global vertex number of the hexahedron's first vertex
/// @param[in] v2 The global vertex number of the hexahedron's second vertex
/// @param[in] v3 The global vertex number of the hexahedron's third vertex
/// @param[in] v4 The global vertex number of the hexahedron's fourth vertex
/// @param[in] v5 The global vertex number of the hexahedron's fifth vertex
/// @param[in] v6 The global vertex number of the hexahedron's sixth vertex
/// @param[in] v7 The global vertex number of the hexahedron's seventh vertex
/// @param[in] v8 The global vertex number of the hexahedron's eighth vertex
/// @return The rotation and reflection orders for the hexahedron
std::array<int, 4> calculate_hexahedron_orders(int v1, int v2, int v3, int v4,
                                               int v5, int v6, int v7, int v8)
{
  if (v1 < v2 and v1 < v3 and v1 < v4 and v1 < v5 and v1 < v6 and v1 < v7 and v1 < v8)
  {
    const std::array<int, 2> tri_orders = calculate_triangle_orders(v2, v3, v5);
    return {0, 0, tri_orders[0], tri_orders[1]};
  }
  else if (v2 < v1 and v2 < v3 and v2 < v4 and v2 < v5 and v2 < v6 and v2 < v7 and v2 < v8)
  {
    const std::array<int, 2> tri_orders = calculate_triangle_orders(v4, v1, v6);
    return {1, 0, tri_orders[0], tri_orders[1]};
  }
  else if (v3 < v1 and v3 < v2 and v3 < v4 and v3 < v5 and v3 < v6 and v3 < v7 and v3 < v8)
  {
    const std::array<int, 2> tri_orders = calculate_triangle_orders(v1, v4, v7);
    return {3, 0, tri_orders[0], tri_orders[1]};
  }
  else if (v4 < v1 and v4 < v2 and v4 < v3 and v4 < v5 and v4 < v6 and v4 < v7 and v4 < v8)
  {
    const std::array<int, 2> tri_orders = calculate_triangle_orders(v2, v3, v8);
    return {2, 0, tri_orders[0], tri_orders[1]};
  }
  else if (v5 < v1 and v5 < v2 and v5 < v3 and v5 < v4 and v5 < v6 and v5 < v7 and v5 < v8)
  {
    const std::array<int, 2> tri_orders = calculate_triangle_orders(v1, v7, v6);
    return {0, 1, tri_orders[0], tri_orders[1]};
  }
  else if (v6 < v1 and v6 < v2 and v6 < v3 and v6 < v4 and v6 < v5 and v6 < v7 and v6 < v8)
  {
    const std::array<int, 2> tri_orders = calculate_triangle_orders(v5, v8, v2);
    return {0, 2, tri_orders[0], tri_orders[1]};
  }
  else if (v7 < v1 and v7 < v2 and v7 < v3 and v7 < v4 and v7 < v5 and v7 < v6 and v7 < v8)
  {
    const std::array<int, 2> tri_orders = calculate_triangle_orders(v8, v5, v3);
    return {2, 2, tri_orders[0], tri_orders[1]};
  }
  else if (v8 < v1 and v8 < v2 and v8 < v3 and v8 < v4 and v8 < v5 and v8 < v6 and v8 < v7)
  {
    const std::array<int, 2> tri_orders = calculate_triangle_orders(v4, v6, v7);
    return {2, 1, tri_orders[0], tri_orders[1]};
  }

  throw std::runtime_error(
      "Two of a hexahedron's vertices appear to be equal.");
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
compute_ordering_triangle(const mesh::Mesh& mesh)
{
  const int num_cells = mesh.num_entities(mesh.topology().dim());
  const int num_permutations = get_num_permutations(mesh.cell_type());
  Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cell_orders(num_cells, num_permutations);

  const std::vector<std::int64_t>& global_indices
      = mesh.topology().global_indices(0);
  // Set orders for each cell
  for (int cell_n = 0; cell_n < num_cells; ++cell_n)
  {
    const mesh::MeshEntity cell(mesh, 2, cell_n);
    const std::int32_t* vertices = cell.entities(0);
    const std::int32_t v0 = global_indices[vertices[0]];
    const std::int32_t v1 = global_indices[vertices[1]];
    const std::int32_t v2 = global_indices[vertices[2]];

    // Set the orders for the edge flips
    cell_orders(cell_n, 0) = (v1 > v2);
    cell_orders(cell_n, 1) = (v0 > v2);
    cell_orders(cell_n, 2) = (v0 > v1);

    // Set the orders for the face rotation and reflection
    const std::array<int, 2> tri_orders = calculate_triangle_orders(v0, v1, v2);
    cell_orders(cell_n, 3) = tri_orders[0];
    cell_orders(cell_n, 4) = tri_orders[1];
  }

  return cell_orders;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
compute_ordering_interval(const mesh::Mesh& mesh)
{
  const int num_cells = mesh.num_entities(mesh.topology().dim());
  const int num_permutations = get_num_permutations(mesh.cell_type());
  Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cell_orders(num_cells, num_permutations);

  // Set orders for each cell
  const std::vector<std::int64_t>& global_indices
      = mesh.topology().global_indices(0);
  for (int cell_n = 0; cell_n < num_cells; ++cell_n)
  {
    const mesh::MeshEntity cell(mesh, 1, cell_n);
    const std::int32_t* vertices = cell.entities(0);
    const std::int32_t v0 = global_indices[vertices[0]];
    const std::int32_t v1 = global_indices[vertices[1]];

    // Set the orders for the edge flip
    cell_orders(cell_n, 0) = (v0 > v1);
  }

  return cell_orders;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
compute_ordering_quadrilateral(const mesh::Mesh& mesh)
{
  const int num_cells = mesh.num_entities(mesh.topology().dim());
  const int num_permutations = get_num_permutations(mesh.cell_type());
  Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cell_orders(num_cells, num_permutations);

  // Set orders for each cell
  const std::vector<std::int64_t>& global_indices
      = mesh.topology().global_indices(0);
  for (int cell_n = 0; cell_n < num_cells; ++cell_n)
  {
    const mesh::MeshEntity cell(mesh, 2, cell_n);
    const std::int32_t* vertices = cell.entities(0);
    const std::int32_t v0 = global_indices[vertices[0]];
    const std::int32_t v1 = global_indices[vertices[1]];
    const std::int32_t v2 = global_indices[vertices[2]];
    const std::int32_t v3 = global_indices[vertices[3]];

    // Set the orders for the edge flips
    cell_orders(cell_n, 0) = (v0 > v1);
    cell_orders(cell_n, 1) = (v2 > v3);
    cell_orders(cell_n, 2) = (v0 > v2);
    cell_orders(cell_n, 3) = (v1 > v3);

    // Set the orders for the face rotation and reflection
    const std::array<int, 2> quad_orders
        = calculate_quadrilateral_orders(v0, v1, v2, v3);
    cell_orders(cell_n, 4) = quad_orders[0];
    cell_orders(cell_n, 5) = quad_orders[1];
  }

  return cell_orders;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
compute_ordering_tetrahedron(const mesh::Mesh& mesh)
{
  const int num_cells = mesh.num_entities(mesh.topology().dim());
  const int num_permutations = get_num_permutations(mesh.cell_type());
  Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cell_orders(num_cells, num_permutations);

  // Set orders for each cell
  const std::vector<std::int64_t>& global_indices
      = mesh.topology().global_indices(0);
  for (int cell_n = 0; cell_n < num_cells; ++cell_n)
  {
    const mesh::MeshEntity cell(mesh, 3, cell_n);
    const std::int32_t* vertices = cell.entities(0);
    const std::int32_t v0 = global_indices[vertices[0]];
    const std::int32_t v1 = global_indices[vertices[1]];
    const std::int32_t v2 = global_indices[vertices[2]];
    const std::int32_t v3 = global_indices[vertices[3]];

    // Set the orders for the edge flips
    cell_orders(cell_n, 0) = (v2 > v3);
    cell_orders(cell_n, 1) = (v1 > v3);
    cell_orders(cell_n, 2) = (v1 > v2);
    cell_orders(cell_n, 3) = (v0 > v3);
    cell_orders(cell_n, 4) = (v0 > v2);
    cell_orders(cell_n, 5) = (v0 > v1);

    // Set the orders for the face rotations and reflections
    const std::array<int, 2> tri_orders0
        = calculate_triangle_orders(v1, v2, v3);
    cell_orders(cell_n, 6) = tri_orders0[0];
    cell_orders(cell_n, 7) = tri_orders0[1];
    const std::array<int, 2> tri_orders1
        = calculate_triangle_orders(v0, v2, v3);
    cell_orders(cell_n, 8) = tri_orders1[0];
    cell_orders(cell_n, 9) = tri_orders1[1];
    const std::array<int, 2> tri_orders2
        = calculate_triangle_orders(v0, v1, v3);
    cell_orders(cell_n, 10) = tri_orders2[0];
    cell_orders(cell_n, 11) = tri_orders2[1];
    const std::array<int, 2> tri_orders3
        = calculate_triangle_orders(v0, v1, v2);
    cell_orders(cell_n, 12) = tri_orders3[0];
    cell_orders(cell_n, 13) = tri_orders3[1];

    // Set the orders for the volume rotations and reflections
    const std::array<int, 4> tet_orders
        = calculate_tetrahedron_orders(v0, v1, v2, v3);
    cell_orders(cell_n, 14) = tet_orders[0];
    cell_orders(cell_n, 15) = tet_orders[1];
    cell_orders(cell_n, 16) = tet_orders[2];
    cell_orders(cell_n, 17) = tet_orders[3];
  }

  return cell_orders;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
compute_ordering_hexahedron(const mesh::Mesh& mesh)
{
  const int num_cells = mesh.num_entities(mesh.topology().dim());
  const int num_permutations = get_num_permutations(mesh.cell_type());
  Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cell_orders(num_cells, num_permutations);

  // Set orders for each cell
  const std::vector<std::int64_t>& global_indices
      = mesh.topology().global_indices(0);
  for (int cell_n = 0; cell_n < num_cells; ++cell_n)
  {
    const mesh::MeshEntity cell(mesh, 3, cell_n);
    const std::int32_t* vertices = cell.entities(0);
    const std::int32_t v0 = global_indices[vertices[0]];
    const std::int32_t v1 = global_indices[vertices[1]];
    const std::int32_t v2 = global_indices[vertices[2]];
    const std::int32_t v3 = global_indices[vertices[3]];
    const std::int32_t v4 = global_indices[vertices[4]];
    const std::int32_t v5 = global_indices[vertices[5]];
    const std::int32_t v6 = global_indices[vertices[6]];
    const std::int32_t v7 = global_indices[vertices[7]];

    // Set the orders for the edge flips
    cell_orders(cell_n, 0) = (v0 > v1);
    cell_orders(cell_n, 1) = (v2 > v3);
    cell_orders(cell_n, 2) = (v4 > v5);
    cell_orders(cell_n, 3) = (v6 > v7);
    cell_orders(cell_n, 4) = (v0 > v2);
    cell_orders(cell_n, 5) = (v1 > v3);
    cell_orders(cell_n, 6) = (v4 > v6);
    cell_orders(cell_n, 7) = (v5 > v7);
    cell_orders(cell_n, 8) = (v0 > v4);
    cell_orders(cell_n, 9) = (v1 > v5);
    cell_orders(cell_n, 10) = (v2 > v6);
    cell_orders(cell_n, 11) = (v3 > v7);

    // Set the orders for the face rotations and reflections
    const std::array<int, 2> quad_orders0
        = calculate_quadrilateral_orders(v0, v1, v2, v3);
    cell_orders(cell_n, 12) = quad_orders0[0];
    cell_orders(cell_n, 13) = quad_orders0[1];
    const std::array<int, 2> quad_orders1
        = calculate_quadrilateral_orders(v4, v5, v6, v7);
    cell_orders(cell_n, 14) = quad_orders1[0];
    cell_orders(cell_n, 15) = quad_orders1[1];
    const std::array<int, 2> quad_orders2
        = calculate_quadrilateral_orders(v0, v1, v4, v5);
    cell_orders(cell_n, 16) = quad_orders2[0];
    cell_orders(cell_n, 17) = quad_orders2[1];
    const std::array<int, 2> quad_orders3
        = calculate_quadrilateral_orders(v2, v3, v6, v7);
    cell_orders(cell_n, 18) = quad_orders3[0];
    cell_orders(cell_n, 19) = quad_orders3[1];
    const std::array<int, 2> quad_orders4
        = calculate_quadrilateral_orders(v0, v2, v4, v6);
    cell_orders(cell_n, 20) = quad_orders4[0];
    cell_orders(cell_n, 21) = quad_orders4[1];
    const std::array<int, 2> quad_orders5
        = calculate_quadrilateral_orders(v1, v3, v5, v7);
    cell_orders(cell_n, 22) = quad_orders5[0];
    cell_orders(cell_n, 23) = quad_orders5[1];

    // Set the orders for the volume rotations and reflections
    const std::array<int, 4> hex_orders
        = calculate_hexahedron_orders(v0, v1, v2, v3, v4, v5, v6, v7);
    cell_orders(cell_n, 24) = hex_orders[0];
    cell_orders(cell_n, 25) = hex_orders[1];
    cell_orders(cell_n, 26) = hex_orders[2];
    cell_orders(cell_n, 27) = hex_orders[3];
  }

  return cell_orders;
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
      cell_ordering;
  switch (mesh.cell_type())
  {
  case (mesh::CellType::point):
    cell_ordering.resize(mesh.num_entities(mesh.topology().dim()), 0);
    break;
  case (mesh::CellType::interval):
    cell_ordering = compute_ordering_interval(mesh);
    break;
  case (mesh::CellType::triangle):
    cell_ordering = compute_ordering_triangle(mesh);
    break;
  case (mesh::CellType::tetrahedron):
    cell_ordering = compute_ordering_tetrahedron(mesh);
    break;
  case (mesh::CellType::quadrilateral):
    cell_ordering = compute_ordering_quadrilateral(mesh);
    break;
  case (mesh::CellType::hexahedron):
    cell_ordering = compute_ordering_hexahedron(mesh);
    break;

  default:
    // The switch should exit before this is reached
    throw std::runtime_error("Unrecognised cell type.");
  }

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
