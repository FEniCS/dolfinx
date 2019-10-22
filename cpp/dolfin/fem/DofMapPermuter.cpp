// Copyright (C) 2019 Matthew Scroggs
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DofMapPermuter.h"
#include <dolfin/common/log.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshIterator.h>

namespace dolfin
{

namespace
{
//-----------------------------------------------------------------------------
void _permute(std::vector<int>& vec,
              Eigen::Array<PetscInt, Eigen::Dynamic, 1> perm)
{
  int temp[vec.size()];
  for (std::size_t i = 0; i < vec.size(); ++i)
    temp[i] = vec[i];
  for (std::size_t i = 0; i < vec.size(); ++i)
    vec[perm[i]] = temp[i];
}
//-----------------------------------------------------------------------------
/// Calculates the permutation orders for a triangle
/// @param[in] v1 The global vertex number of the triangle's first vertex
/// @param[in] v2 The global vertex number of the triangle's second vertex
/// @param[in] v3 The global vertex number of the triangle's third vertex
/// @return The rotation and reflection orders for the triangle
std::array<int, 2> _calculate_triangle_orders(int v1, int v2, int v3)
{
  if (v1 < v2 && v1 < v3)
    return {0, v2 > v3};
  if (v2 < v1 && v2 < v3)
    return {1, v3 > v1};
  if (v3 < v1 && v3 < v2)
    return {2, v1 > v2};
}
//-----------------------------------------------------------------------------
/// Calculates the permutation orders for a tetrahedron
/// @param[in] v1 The global vertex number of the tetrahedron's first vertex
/// @param[in] v2 The global vertex number of the tetrahedron's second vertex
/// @param[in] v3 The global vertex number of the tetrahedron's third vertex
/// @param[in] v4 The global vertex number of the tetrahedron's fourth vertex
/// @return The rotation and reflection orders for the tetrahedron
std::array<int, 4> _calculate_tetrahedron_orders(int v1, int v2, int v3, int v4)
{
  if (v1 < v2 && v1 < v3 && v1 < v4)
  {
    auto tri_orders = _calculate_triangle_orders(v2, v3, v4);
    return {0, 0, tri_orders[0], tri_orders[1]};
  }
  if (v2 < v1 && v2 < v3 && v2 < v4)
  {
    auto tri_orders = _calculate_triangle_orders(v3, v1, v4);
    return {1, 0, tri_orders[0], tri_orders[1]};
  }
  if (v3 < v1 && v3 < v2 && v3 < v4)
  {
    int a;
    int b;
    auto tri_orders = _calculate_triangle_orders(v1, v2, v4);
    return {2, 0, tri_orders[0], tri_orders[1]};
  }
  if (v4 < v1 && v4 < v2 && v4 < v3)
  {
    int a;
    int b;
    auto tri_orders = _calculate_triangle_orders(v2, v1, v3);
    return {0, 1, tri_orders[0], tri_orders[1]};
  }
}
//-----------------------------------------------------------------------------
/// Makes a permutation to flip the dofs on an edge
/// @param[in] edge_dofs Number of edge dofs
/// @param[in] bs Block size
/// @return The permutation to reverse the dofs
std::vector<int> _edge_flip(const int edge_dofs, const int blocksize)
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
_triangle_rotation_and_reflection(const int face_dofs, const int blocksize)
{
  // This will only be called at most once for each mesh
  const int blocks = face_dofs / blocksize;
  float root = std::sqrt(8 * blocks + 1);
  assert(root == std::floor(root) && root % 2 == 1);
  int side_length = (root - 1) / 2; // side length of the triangle of face dofs

  std::vector<int> rotation(face_dofs);
  {
    int j = 0;
    int i = 1;
    for (int st = blocks - 1; st >= 0; st -= (i++))
    {
      int dof = st;
      for (int sub = i + 1; sub <= side_length + 1; dof -= (sub++))
        for (int k = 0; k < blocksize; ++k)
          rotation[j++] = blocksize * dof + k;
    }
    assert(j == face_dofs);
  }

  std::vector<int> reflection(face_dofs);
  {
    int j = 0;
    for (int st = 0; st < side_length; ++st)
    {
      int dof = st;
      for (int add = side_length; add > st; dof += (add--))
        for (int k = 0; k < blocksize; ++k)
          reflection[j++] = blocksize * dof + k;
    }
    assert(j == face_dofs);
  }

  return {rotation, reflection};
}
//-----------------------------------------------------------------------------
/// Makes permutations to rotate and reflect the dofs in a tetrahedron
/// @param[in] edge_dofs Number of dofs on the interior of the tetrahedron
/// @return Permutations to rotate and reflect the dofs
std::array<std::vector<int>, 4>
_tetrahedron_rotations_and_reflection(const int volume_dofs,
                                      const int blocksize)
{
  // This will only be called at most once for each mesh
  const int blocks = volume_dofs / blocksize;
  int side_length = 0;
  while (side_length * (side_length + 1) * (side_length + 2) < 6 * blocks)
    ++side_length;
  assert(side_length * (side_length + 1) * (side_length + 2) == 6 * blocks);

  std::vector<int> rotation1(volume_dofs);
  std::iota(rotation1.begin(), rotation1.end(), 0);
  std::vector<int> reflection(volume_dofs);
  std::iota(reflection.begin(), reflection.end(), 0);
  std::vector<int> rotation2(volume_dofs);
  std::iota(rotation2.begin(), rotation2.end(), 0);

  int start = 0;
  for (int side = side_length; side > 0; --side)
  {
    int face_dofs = side * (side + 1) / 2;
    auto base_faces = _triangle_rotation_and_reflection(face_dofs, 1);

    std::vector<int> face(face_dofs * blocksize);
    std::iota(face.begin(), face.end(), start);

    std::vector<int> face2(face_dofs * blocksize);
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
      reflection[face[j]] = face[base_faces[1][j]];
    }
  }

  std::vector<int> rotation3(volume_dofs);
  for (int j = 0; j < volume_dofs; ++j)
    rotation3[j] = rotation2[rotation2[rotation1[j]]];
  return {rotation1, rotation2, rotation3, reflection};
}
//-----------------------------------------------------------------------------
} // namespace

namespace fem
{
//-----------------------------------------------------------------------------
// public:
//-----------------------------------------------------------------------------
DofMapPermuter::DofMapPermuter(const mesh::Mesh mesh,
                               const ElementDofLayout& element_dof_layout)
{
  const mesh::CellType type = mesh.cell_type();
  switch (type)
  {
  case (mesh::CellType::triangle):
    _generate_triangle(mesh, element_dof_layout);
    return;
  case (mesh::CellType::tetrahedron):
    _generate_tetrahedron(mesh, element_dof_layout);
    return;
  default:
    LOG(WARNING) << "Dof permutations are not defined for this cell type. High "
                    "order elements may be incorrect.";
    _generate_empty(mesh, element_dof_layout);
  }
}
//-----------------------------------------------------------------------------
std::vector<int> DofMapPermuter::cell_permutation(const int cell) const
{
  std::vector<int> permutation(_dof_count);
  std::iota(permutation.begin(), permutation.end(), 0);

  for (int i = 0; i < _permutation_count; ++i)
    for (int j = 0; j < _cell_orders(cell, i); ++j)
      _permute(permutation, _permutations.row(i));

  return permutation;
}
//-----------------------------------------------------------------------------
// private:
//-----------------------------------------------------------------------------
void DofMapPermuter::_set_permutation(
    const int index, const Eigen::Array<PetscInt, Eigen::Dynamic, 1> dofs,
    const std::vector<int> base_permutation, int order)
{
  for (std::size_t i = 0; i < base_permutation.size(); ++i)
    _permutations(index, dofs(i)) = dofs(base_permutation[i]);
  _permutation_orders[index] = order;
}
//-----------------------------------------------------------------------------
void DofMapPermuter::_set_order(const int cell, const int permutation,
                                const int order)
{
  _cell_orders(cell, permutation) = order;
}
//-----------------------------------------------------------------------------
void DofMapPermuter::_resize_data()
{
  if (_permutation_count > 0)
  {
    _cell_orders.resize(_cell_count, _permutation_count);
    _cell_orders.fill(0);
    _permutations.resize(_permutation_count, _dof_count);
    for (int i = 0; i < _dof_count; ++i)
      for (int j = 0; j < _permutation_count; ++j)
        _permutations(j, i) = i;
    _permutation_orders.resize(_permutation_count, 0);
  }
}
//-----------------------------------------------------------------------------
void DofMapPermuter::_generate_triangle(
    const mesh::Mesh mesh, const ElementDofLayout& element_dof_layout)
{
  const int D = mesh.topology().dim();
  const int vertex_dofs = 0 <= D ? element_dof_layout.num_entity_dofs(0) : 0;
  const int edge_dofs = 1 <= D ? element_dof_layout.num_entity_dofs(1) : 0;
  const int face_dofs = 2 <= D ? element_dof_layout.num_entity_dofs(2) : 0;
  const int volume_dofs = 3 <= D ? element_dof_layout.num_entity_dofs(3) : 0;
  const int bs = element_dof_layout.block_size();

  _dof_count = 3 * vertex_dofs + 3 * edge_dofs + face_dofs;
  _permutation_count = 5;
  _cell_count = mesh.num_entities(mesh.topology().dim());

  _resize_data();

  // Make edge flipping permutations
  std::vector<int> base_flip = _edge_flip(edge_dofs, bs);
  for (int edge_n = 0; edge_n < 3; ++edge_n)
  {
    auto edge = element_dof_layout.entity_dofs(1, edge_n);
    _set_permutation(edge_n, edge, base_flip, 2);
  }
  // Make permutations that rotate and reflect the face dofs
  auto base_faces = _triangle_rotation_and_reflection(face_dofs, bs);
  auto face = element_dof_layout.entity_dofs(2, 0);
  _set_permutation(3, face, base_faces[0], 3);
  _set_permutation(4, face, base_faces[1], 2);

  // Set orders for each cell
  for (int cell_n = 0; cell_n < _cell_count; ++cell_n)
  {
    const mesh::MeshEntity cell(mesh, 2, cell_n);
    const std::int32_t* vertices = cell.entities(0);
    std::vector<int> orders(5);
    _set_order(cell_n, 0, vertices[1] > vertices[2]);
    _set_order(cell_n, 1, vertices[0] > vertices[2]);
    _set_order(cell_n, 2, vertices[0] > vertices[1]);

    auto tri_orders
        = _calculate_triangle_orders(vertices[0], vertices[1], vertices[2]);
    _set_order(cell_n, 3, tri_orders[0]);
    _set_order(cell_n, 4, tri_orders[1]);
  }
}
//-----------------------------------------------------------------------------
void DofMapPermuter::_generate_tetrahedron(
    const mesh::Mesh mesh, const ElementDofLayout& element_dof_layout)
{
  const int D = mesh.topology().dim();
  const int vertex_dofs = 0 <= D ? element_dof_layout.num_entity_dofs(0) : 0;
  const int edge_dofs = 1 <= D ? element_dof_layout.num_entity_dofs(1) : 0;
  const int face_dofs = 2 <= D ? element_dof_layout.num_entity_dofs(2) : 0;
  const int volume_dofs = 3 <= D ? element_dof_layout.num_entity_dofs(3) : 0;
  const int bs = element_dof_layout.block_size();

  _dof_count = 4 * vertex_dofs + 6 * edge_dofs + 4 * face_dofs + volume_dofs;
  _permutation_count = 18;
  _cell_count = mesh.num_entities(mesh.topology().dim());

  _resize_data();

  // Make edge flipping permutations
  std::vector<int> base_flip = _edge_flip(edge_dofs, bs);
  for (int edge_n = 0; edge_n < 6; ++edge_n)
  {
    auto edge = element_dof_layout.entity_dofs(1, edge_n);
    _set_permutation(edge_n, edge, base_flip, 2);
  }
  // Make permutations that rotate and reflect the face dofs
  auto base_faces = _triangle_rotation_and_reflection(face_dofs, bs);
  for (int face_n = 0; face_n < 4; ++face_n)
  {
    auto face = element_dof_layout.entity_dofs(2, face_n);
    _set_permutation(6 + 2 * face_n, face, base_faces[0], 3);
    _set_permutation(6 + 2 * face_n + 1, face, base_faces[1], 2);
  }
  // Make permutations that rotate and reflect the interior dofs
  auto base_interiors = _tetrahedron_rotations_and_reflection(volume_dofs, bs);
  auto interior = element_dof_layout.entity_dofs(3, 0);
  _set_permutation(14, interior, base_interiors[0], 3);
  _set_permutation(15, interior, base_interiors[1], 3);
  _set_permutation(16, interior, base_interiors[2], 3);
  _set_permutation(17, interior, base_interiors[3], 2);

  // Set orders for each cell
  for (int cell_n = 0; cell_n < _cell_count; ++cell_n)
  {
    const mesh::MeshEntity cell(mesh, 3, cell_n);
    const std::int32_t* vertices = cell.entities(0);

    _set_order(cell_n, 0, vertices[2] > vertices[3]);
    _set_order(cell_n, 1, vertices[1] > vertices[3]);
    _set_order(cell_n, 2, vertices[1] > vertices[2]);
    _set_order(cell_n, 3, vertices[0] > vertices[3]);
    _set_order(cell_n, 4, vertices[0] > vertices[2]);
    _set_order(cell_n, 5, vertices[0] > vertices[1]);

    auto tri_orders
        = _calculate_triangle_orders(vertices[1], vertices[2], vertices[3]);
    _set_order(cell_n, 6, tri_orders[0]);
    _set_order(cell_n, 7, tri_orders[1]);
    tri_orders
        = _calculate_triangle_orders(vertices[0], vertices[2], vertices[3]);
    _set_order(cell_n, 8, tri_orders[0]);
    _set_order(cell_n, 9, tri_orders[1]);
    tri_orders
        = _calculate_triangle_orders(vertices[0], vertices[1], vertices[3]);
    _set_order(cell_n, 10, tri_orders[0]);
    _set_order(cell_n, 11, tri_orders[1]);
    tri_orders
        = _calculate_triangle_orders(vertices[0], vertices[1], vertices[2]);
    _set_order(cell_n, 12, tri_orders[0]);
    _set_order(cell_n, 13, tri_orders[1]);

    auto tet_orders = _calculate_tetrahedron_orders(vertices[0], vertices[1],
                                                    vertices[2], vertices[3]);
    _set_order(cell_n, 14, tet_orders[0]);
    _set_order(cell_n, 15, tet_orders[1]);
    _set_order(cell_n, 16, tet_orders[2]);
    _set_order(cell_n, 17, tet_orders[3]);
  }
}
//-----------------------------------------------------------------------------
void DofMapPermuter::_generate_empty(const mesh::Mesh mesh,
                                     const ElementDofLayout& element_dof_layout)
{
  _dof_count = element_dof_layout.num_dofs();
  _cell_count = mesh.num_entities(mesh.topology().dim());
  _permutation_count = 0;
  _resize_data();
}
//-----------------------------------------------------------------------------
} // namespace fem
} // namespace dolfin
