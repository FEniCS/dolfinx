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
  int* temp = new int[vec.size()];
  for (std::size_t i = 0; i < vec.size(); ++i)
    temp[i] = vec[i];
  for (std::size_t i = 0; i < vec.size(); ++i)
    vec[perm[i]] = temp[i];
  delete[] temp;
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
  throw std::runtime_error("Two of a triangle's vertices appear to be equal.");
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
    auto tri_orders = _calculate_triangle_orders(v1, v2, v4);
    return {2, 0, tri_orders[0], tri_orders[1]};
  }
  if (v4 < v1 && v4 < v2 && v4 < v3)
  {
    auto tri_orders = _calculate_triangle_orders(v2, v1, v3);
    return {0, 1, tri_orders[0], tri_orders[1]};
  }
  throw std::runtime_error(
      "Two of a tetrahedron's vertices appear to be equal.");
}
//-----------------------------------------------------------------------------
/// Makes a permutation to flip the dofs on an edge
/// @param[in] edge_dofs Number of edge dofs
/// @return The permutation to reverse the dofs
std::vector<int> _edge_flip(const int edge_dofs)
{
  // This will only be called at most once for each mesh
  int j = 0;
  std::vector<int> flip(edge_dofs);
  for (int i = 0; i < edge_dofs; ++i)
    flip[j++] = edge_dofs - (i + 1);
  return flip;
}
//-----------------------------------------------------------------------------
/// Makes permutations to rotate and reflect the dofs on a triangle
/// @param[in] edge_dofs Number of dofs on the face of the triangle
/// @return Permutations to rotate and reflect the dofs
std::array<std::vector<int>, 2>
_triangle_rotation_and_reflection(const int face_dofs)
{
  // This will only be called at most once for each mesh
  float root = std::sqrt(8 * face_dofs + 1);
  assert(root == std::floor(root) && int(root) % 2 == 1);
  int side_length = (root - 1) / 2; // side length of the triangle of face dofs

  std::vector<int> rotation(face_dofs);
  {
    int j = 0;
    int i = 1;
    for (int st = face_dofs - 1; st >= 0; st -= (i++))
    {
      int dof = st;
      for (int sub = i + 1; sub <= side_length + 1; dof -= (sub++))
        rotation[j++] = dof;
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
        reflection[j++] = dof;
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
_tetrahedron_rotations_and_reflection(const int volume_dofs)
{
  // This will only be called at most once for each mesh
  int side_length = 0;
  while (side_length * (side_length + 1) * (side_length + 2) < 6 * volume_dofs)
    ++side_length;
  assert(side_length * (side_length + 1) * (side_length + 2)
         == 6 * volume_dofs);

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
    auto base_faces = _triangle_rotation_and_reflection(face_dofs);

    std::vector<int> face(face_dofs);
    std::iota(face.begin(), face.end(), start);

    std::vector<int> face2(face_dofs);
    int j = 0;
    int start2 = side * side_length - 1 - side * (side - 1) / 2;
    for (int row = 0; row < side; ++row)
    {
      int dof = start2;
      face2[j++] = dof;
      for (int sub = 2 + (side_length - side); sub <= side_length - row; ++sub)
      {
        dof -= sub;
        face2[j++] = dof;
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
void _apply_permutation(
    Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic>& permutations,
    const int index, const Eigen::Array<PetscInt, Eigen::Dynamic, 1> dofs,
    const std::vector<int> base_permutation, int order)
{
  for (std::size_t i = 0; i < base_permutation.size(); ++i)
    permutations(index, dofs(i)) = dofs(base_permutation[i]);
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
  _dof_count = element_dof_layout.num_dofs();
  _cell_count = mesh.num_entities(mesh.topology().dim());
  switch (mesh.cell_type())
  {
  case (mesh::CellType::triangle):
    _permutation_count = 5;
    break;
  case (mesh::CellType::tetrahedron):
    _permutation_count = 18;
    break;
  default:
    _permutation_count = 0;
    LOG(WARNING) << "Dof permutations are not defined for this cell type. High "
                    "order elements may be incorrect.";
    return;
  }
  _resize_data();
  _permutations = _generate_recursive(mesh, element_dof_layout);
  _set_orders(mesh, element_dof_layout);
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
Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic>
DofMapPermuter::_generate_recursive(const mesh::Mesh mesh,
                                    const ElementDofLayout& element_dof_layout)
{
  if (element_dof_layout.num_sub_dofmaps() == 0)
  {
    switch (mesh.cell_type())
    {
    case (mesh::CellType::triangle):
      return _generate_triangle(mesh, element_dof_layout);
    case (mesh::CellType::tetrahedron):
      return _generate_tetrahedron(mesh, element_dof_layout);
    default:
      throw std::runtime_error(
          "Unrecognised cell type."); // The function should exit before this is
                                      // reached
    }
  }

  Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic> output(
      _permutation_count, element_dof_layout.num_dofs());

  for (int i = 0; i < element_dof_layout.num_sub_dofmaps(); ++i)
  {
    auto sub_view = element_dof_layout.sub_view({i});
    auto sub_perm
        = _generate_recursive(mesh, *element_dof_layout.sub_dofmap({i}));
    for (int p = 0; p < _permutation_count; ++p)
      for (std::size_t j = 0; j < sub_view.size(); ++j)
        output(p, sub_view[j]) = sub_view[sub_perm(p, j)];
  }
  return output;
}
//-----------------------------------------------------------------------------
void DofMapPermuter::_set_orders(const mesh::Mesh mesh,
                                 const ElementDofLayout& element_dof_layout)
{
  switch (mesh.cell_type())
  {
  case (mesh::CellType::triangle):
    _set_orders_triangle(mesh, element_dof_layout);
    return;
  case (mesh::CellType::tetrahedron):
    _set_orders_tetrahedron(mesh, element_dof_layout);
    return;
  default:
    throw std::runtime_error(
        "Unrecognised cell type."); // The function should exit before this is
                                    // reached
  }
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
  _cell_orders.resize(_cell_count, _permutation_count);
  _cell_orders.fill(0);
  _permutations.resize(_permutation_count, _dof_count);
  for (int i = 0; i < _dof_count; ++i)
    for (int j = 0; j < _permutation_count; ++j)
      _permutations(j, i) = i;
}
//-----------------------------------------------------------------------------
Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic>
DofMapPermuter::_generate_triangle(const mesh::Mesh mesh,
                                   const ElementDofLayout& element_dof_layout)
{
  // TODO: Make this return an Array, then make the recursive function put it
  // into the higher level array
  const int D = mesh.topology().dim();
  const int edge_dofs = 1 <= D ? element_dof_layout.num_entity_dofs(1) : 0;
  const int face_dofs = 2 <= D ? element_dof_layout.num_entity_dofs(2) : 0;

  Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic> permutations(
      _permutation_count, _dof_count);
  for (int i = 0; i < element_dof_layout.num_dofs(); ++i)
    for (int j = 0; j < _permutation_count; ++j)
      permutations(j, i) = i;

  // Make edge flipping permutations
  std::vector<int> base_flip = _edge_flip(edge_dofs);
  for (int edge_n = 0; edge_n < 3; ++edge_n)
  {
    auto edge = element_dof_layout.entity_dofs(1, edge_n);
    _apply_permutation(permutations, edge_n, edge, base_flip, 2);
  }
  // Make permutations that rotate and reflect the face dofs
  auto base_faces = _triangle_rotation_and_reflection(face_dofs);
  auto face = element_dof_layout.entity_dofs(2, 0);
  _apply_permutation(permutations, 3, face, base_faces[0], 3);
  _apply_permutation(permutations, 4, face, base_faces[1], 2);
  return permutations;
}
//-----------------------------------------------------------------------------
Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic>
DofMapPermuter::_generate_tetrahedron(
    const mesh::Mesh mesh, const ElementDofLayout& element_dof_layout)
{
  // TODO: Make this return an Array, then make the recursive function put it
  // into the higher level array
  const int D = mesh.topology().dim();
  const int edge_dofs = 1 <= D ? element_dof_layout.num_entity_dofs(1) : 0;
  const int face_dofs = 2 <= D ? element_dof_layout.num_entity_dofs(2) : 0;
  const int volume_dofs = 3 <= D ? element_dof_layout.num_entity_dofs(3) : 0;

  Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic> permutations(
      _permutation_count, _dof_count);
  for (int i = 0; i < element_dof_layout.num_dofs(); ++i)
    for (int j = 0; j < _permutation_count; ++j)
      permutations(j, i) = i;

  // Make edge flipping permutations
  std::vector<int> base_flip = _edge_flip(edge_dofs);
  for (int edge_n = 0; edge_n < 6; ++edge_n)
  {
    auto edge = element_dof_layout.entity_dofs(1, edge_n);
    _apply_permutation(permutations, edge_n, edge, base_flip, 2);
  }
  // Make permutations that rotate and reflect the face dofs
  auto base_faces = _triangle_rotation_and_reflection(face_dofs);
  for (int face_n = 0; face_n < 4; ++face_n)
  {
    auto face = element_dof_layout.entity_dofs(2, face_n);
    _apply_permutation(permutations, 6 + 2 * face_n, face, base_faces[0], 3);
    _apply_permutation(permutations, 6 + 2 * face_n + 1, face, base_faces[1],
                       2);
  }
  // Make permutations that rotate and reflect the interior dofs
  auto base_interiors = _tetrahedron_rotations_and_reflection(volume_dofs);
  auto interior = element_dof_layout.entity_dofs(3, 0);
  _apply_permutation(permutations, 14, interior, base_interiors[0], 3);
  _apply_permutation(permutations, 15, interior, base_interiors[1], 3);
  _apply_permutation(permutations, 16, interior, base_interiors[2], 3);
  _apply_permutation(permutations, 17, interior, base_interiors[3], 2);
  return permutations;
}
//-----------------------------------------------------------------------------
void DofMapPermuter::_set_orders_triangle(
    const mesh::Mesh mesh, const ElementDofLayout& element_dof_layout)
{
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
void DofMapPermuter::_set_orders_tetrahedron(
    const mesh::Mesh mesh, const ElementDofLayout& element_dof_layout)
{
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
} // namespace fem
} // namespace dolfin
