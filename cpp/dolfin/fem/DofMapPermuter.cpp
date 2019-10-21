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

namespace fem
{
//-----------------------------------------------------------------------------
DofMapPermuter::DofMapPermuter(){};
//-----------------------------------------------------------------------------
void DofMapPermuter::set_dof_count(const int dofs) { _dof_count = dofs; }
//-----------------------------------------------------------------------------
void DofMapPermuter::add_permutation(const std::vector<int> permutation, int order)
{
  _permutations.push_back(permutation);
  _permutation_orders.push_back(order);
}
//-----------------------------------------------------------------------------
void DofMapPermuter::set_cell(const int cell, const std::vector<int> orders)
{
  for (std::size_t i = 0; i < orders.size(); ++i)
    _cell_orders(cell, i) = orders[i];
}
//-----------------------------------------------------------------------------
void DofMapPermuter::set_cell_count(const int cells)
{
  _cell_count = cells;
  if (_permutations.size() > 0)
  {
    _cell_orders.resize(_cell_count, _permutations.size());
    _cell_orders.fill(0);
  }
}
//-----------------------------------------------------------------------------
std::vector<int> DofMapPermuter::permute(std::vector<int> vec, std::vector<int> perm) const
{
  std::vector<int> output(perm.size());
  for (std::size_t i = 0; i < perm.size(); ++i)
    output[perm[i]] = vec[i];
  return output;
}
//-----------------------------------------------------------------------------
std::vector<int> DofMapPermuter::cell_permutation(const int cell) const
{
  std::vector<int> permutation(_dof_count);
  std::iota(permutation.begin(), permutation.end(), 0);

  if (_permutations.size() > 0)
    for (std::size_t i = 0; i < _permutations.size(); ++i)
      for (int j = 0; j < _cell_orders(cell, i); ++j)
        permutation = permute(permutation, _permutations[i]);

  return permutation;
}
//-----------------------------------------------------------------------------
/// Makes a permutation to flip the dofs on an edge
/// @param[in] edge_dofs Number of edge dofs
/// @return The permutation to reverse the dofs
std::vector<int> edge_flip(const int edge_dofs)
{
  std::vector<int> flip(edge_dofs);
  for (int i = 0; i < edge_dofs; ++i)
    flip[i] = edge_dofs - 1 - i;
  return flip;
}
//-----------------------------------------------------------------------------
/// Makes permutations to rotate and reflect the dofs on a triangle
/// @param[in] edge_dofs Number of dofs on the face of the triangle
/// @return Permutations to rotate and reflect the dofs
std::pair<std::vector<int>, std::vector<int>>
triangle_rotation_and_reflection(const int face_dofs)
{
  float root = std::sqrt(8 * face_dofs + 1);
  assert(root == floor(root) && root % 2 == 1);
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

  return std::make_pair(rotation, reflection);
}
//-----------------------------------------------------------------------------
/// Makes permutations to rotate and reflect the dofs in a tetrahedron
/// @param[in] edge_dofs Number of dofs on the interior of the tetrahedron
/// @return Permutations to rotate and reflect the dofs
std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
tetrahedron_rotations_and_reflection(const int volume_dofs)
{
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
    std::vector<int> base_face_rotation;
    std::vector<int> base_face_reflection;
    tie(base_face_rotation, base_face_reflection)
        = triangle_rotation_and_reflection(face_dofs);

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
      rotation1[face[j]] = face[base_face_rotation[j]];
      rotation2[face2[j]] = face2[base_face_rotation[j]];
      reflection[face[j]] = face[base_face_reflection[j]];
    }
  }

  return std::make_tuple(rotation1, rotation2, reflection);
}
//-----------------------------------------------------------------------------
std::pair<int, int> DofMapPermuter::calculate_triangle_orders(int v1, int v2,
                                                              int v3)
{
  if (v1 < v2 && v1 < v3)
    return std::make_pair(0, v2 > v3);
  if (v2 < v1 && v2 < v3)
    return std::make_pair(1, v3 > v1);
  if (v3 < v1 && v3 < v2)
    return std::make_pair(2, v1 > v2);
}
//-----------------------------------------------------------------------------
std::array<int, 4> DofMapPermuter::calculate_tetrahedron_orders(int v1, int v2,
                                                                int v3, int v4)
{
  if (v1 < v2 && v1 < v3 && v1 < v4)
  {
    int a;
    int b;
    std::tie(a, b) = calculate_triangle_orders(v2, v3, v4);
    return {0, 0, a, b};
  }
  if (v2 < v1 && v2 < v3 && v2 < v4)
  {
    int a;
    int b;
    std::tie(a, b) = calculate_triangle_orders(v3, v1, v4);
    return {1, 0, a, b};
  }
  if (v3 < v1 && v3 < v2 && v3 < v4)
  {
    int a;
    int b;
    std::tie(a, b) = calculate_triangle_orders(v1, v2, v4);
    return {2, 0, a, b};
  }
  if (v4 < v1 && v4 < v2 && v4 < v3)
  {
    int a;
    int b;
    std::tie(a, b) = calculate_triangle_orders(v2, v1, v3);
    return {0, 1, a, b};
  }
}
//-----------------------------------------------------------------------------
/// Make the DofMapPermuter for a given triangular mesh and dof layout
/// @param[in] mesh The mesh
/// @param[in] element_dof_layout The layout of dofs in each cell
/// @return A DofMapPermuter for the mesh and dof layout
DofMapPermuter
_generate_cell_permutations_triangle(const mesh::Mesh mesh,
                                     const ElementDofLayout& element_dof_layout)
{
  const int D = mesh.topology().dim();
  const int vertex_dofs = 0 <= D ? element_dof_layout.num_entity_dofs(0) : 0;
  const int edge_dofs = 1 <= D ? element_dof_layout.num_entity_dofs(1) : 0;
  const int face_dofs = 2 <= D ? element_dof_layout.num_entity_dofs(2) : 0;
  const int volume_dofs = 3 <= D ? element_dof_layout.num_entity_dofs(3) : 0;

  const int dof_count = 3*vertex_dofs + 3*edge_dofs + face_dofs;
  DofMapPermuter permuter;
  permuter.set_dof_count(dof_count);

  float root = std::sqrt(8*face_dofs+1);
  assert(root == floor(root) && root%2 == 1);
  int side_length = (root-1)/2; // side length of the triangle of face dofs

  // Make edge flipping permutations
  std::vector<int> base_flip = edge_flip(edge_dofs);
  for (int edge_n = 0; edge_n < 3; ++edge_n)
  {
    std::vector<int> flip(dof_count);
    std::iota(flip.begin(), flip.end(), 0);
    auto edge = element_dof_layout.entity_dofs(1, edge_n);
    for (std::size_t j = 0; j < edge.size(); ++j)
      flip[edge[j]] = edge[base_flip[j]];
    permuter.add_permutation(flip, 2);
  }

  std::vector<int> base_face_rotation;
  std::vector<int> base_face_reflection;
  tie(base_face_rotation, base_face_reflection)
      = triangle_rotation_and_reflection(face_dofs);
  // Make permutation that rotates the face dofs
  {
    std::vector<int> rotation(dof_count);
    std::iota(rotation.begin(), rotation.end(), 0);
    auto face = element_dof_layout.entity_dofs(2, 0);
    for (std::size_t j = 0; j < face.size(); ++j)
      rotation[face[j]] = face[base_face_rotation[j]];
    permuter.add_permutation(rotation, 3);
  }

  // Make permutation that reflects the face dofs
  {
    std::vector<int> reflection(dof_count);
    std::iota(reflection.begin(), reflection.end(), 0);
    auto face = element_dof_layout.entity_dofs(2, 0);
    for (std::size_t j = 0; j < face.size(); ++j)
      reflection[face[j]] = face[base_face_reflection[j]];
    permuter.add_permutation(reflection, 2);
  }

  int cells = mesh.num_entities(mesh.topology().dim());
  permuter.set_cell_count(cells);

  for (int cell_n = 0; cell_n < cells; ++cell_n)
  {
    const mesh::MeshEntity cell(mesh, 2, cell_n);
    const std::int32_t* vertices = cell.entities(0);
    std::vector<int> orders(5);
    orders[0] = (vertices[1] > vertices[2]);
    orders[1] = (vertices[0] > vertices[2]);
    orders[2] = (vertices[0] > vertices[1]);

    std::tie(orders[3], orders[4]) = permuter.calculate_triangle_orders(
        vertices[0], vertices[1], vertices[2]);

    permuter.set_cell(cell_n, orders);
  }

  return permuter;
}
//-----------------------------------------------------------------------------
/// Make the DofMapPermuter for a given tetrahedral mesh and dof layout
/// @param[in] mesh The mesh
/// @param[in] element_dof_layout The layout of dofs in each cell
/// @return A DofMapPermuter for the mesh and dof layout
DofMapPermuter _generate_cell_permutations_tetrahedron(
    const mesh::Mesh mesh, const ElementDofLayout& element_dof_layout)
{
  const int D = mesh.topology().dim();
  const int vertex_dofs = 0 <= D ? element_dof_layout.num_entity_dofs(0) : 0;
  const int edge_dofs = 1 <= D ? element_dof_layout.num_entity_dofs(1) : 0;
  const int face_dofs = 2 <= D ? element_dof_layout.num_entity_dofs(2) : 0;
  const int volume_dofs = 3 <= D ? element_dof_layout.num_entity_dofs(3) : 0;

  const int dof_count
      = 4 * vertex_dofs + 6 * edge_dofs + 4 * face_dofs + volume_dofs;
  DofMapPermuter permuter;
  permuter.set_dof_count(dof_count);

  float root = std::sqrt(8 * face_dofs + 1);
  assert(root == floor(root) && root % 2 == 1);
  int face_side_length
      = (root - 1) / 2; // side length of the triangle of face dofs

  std::vector<int> base_flip = edge_flip(edge_dofs);

  // Make edge flipping permutations
  for (int edge_n = 0; edge_n < 6; ++edge_n)
  {
    std::vector<int> flip(dof_count);
    std::iota(flip.begin(), flip.end(), 0);
    auto edge = element_dof_layout.entity_dofs(1, edge_n);
    for (std::size_t j = 0; j < edge.size(); ++j)
      flip[edge[j]] = edge[base_flip[j]];
    permuter.add_permutation(flip, 2);
  }

  std::vector<int> base_face_rotation;
  std::vector<int> base_face_reflection;
  tie(base_face_rotation, base_face_reflection)
      = triangle_rotation_and_reflection(face_dofs);

  // Make permutations that rotate the face dofs
  for (int face_n = 0; face_n < 4; ++face_n)
  {
    std::vector<int> rotation(dof_count);
    std::iota(rotation.begin(), rotation.end(), 0);
    auto face = element_dof_layout.entity_dofs(2, face_n);
    for (std::size_t j = 0; j < face.size(); ++j)
      rotation[face[j]] = face[base_face_rotation[j]];
    permuter.add_permutation(rotation, 3);
  }

  // Make permutations that reflect the face dofs
  for (int face_n = 0; face_n < 4; ++face_n)
  {
    std::vector<int> reflection(dof_count);
    std::iota(reflection.begin(), reflection.end(), 0);
    auto face = element_dof_layout.entity_dofs(2, face_n);
    for (std::size_t j = 0; j < face.size(); ++j)
      reflection[face[j]] = face[base_face_reflection[j]];
    permuter.add_permutation(reflection, 2);
  }

  std::vector<int> base_interior_rotation1;
  std::vector<int> base_interior_rotation2;
  std::vector<int> base_interior_reflection;
  tie(base_interior_rotation1, base_interior_rotation2,
      base_interior_reflection)
      = tetrahedron_rotations_and_reflection(volume_dofs);
  auto interior = element_dof_layout.entity_dofs(3, 0);
  {
    std::vector<int> rotation(dof_count);
    std::iota(rotation.begin(), rotation.end(), 0);
    for (std::size_t j = 0; j < interior.size(); ++j)
      rotation[interior[j]] = interior[base_interior_rotation1[j]];
    permuter.add_permutation(rotation, 3);
  }
  {
    std::vector<int> rotation(dof_count);
    std::iota(rotation.begin(), rotation.end(), 0);
    for (std::size_t j = 0; j < interior.size(); ++j)
      rotation[interior[j]] = interior[base_interior_rotation2[j]];
    permuter.add_permutation(rotation, 3);
  }
  {
    std::vector<int> rotation(dof_count);
    std::iota(rotation.begin(), rotation.end(), 0);
    for (std::size_t j = 0; j < interior.size(); ++j)
      rotation[interior[j]]
          = interior[base_interior_rotation2
                         [base_interior_rotation2[base_interior_rotation1[j]]]];
    permuter.add_permutation(rotation, 3);
  }
  {
    std::vector<int> reflection(dof_count);
    std::iota(reflection.begin(), reflection.end(), 0);
    for (std::size_t j = 0; j < interior.size(); ++j)
      reflection[interior[j]] = interior[base_interior_reflection[j]];
    permuter.add_permutation(reflection, 2);
  }

  int cells = mesh.num_entities(mesh.topology().dim());
  permuter.set_cell_count(cells);

  for (int cell_n = 0; cell_n < cells; ++cell_n)
  {
    const mesh::MeshEntity cell(mesh, 3, cell_n);
    const std::int32_t* vertices = cell.entities(0);
    std::vector<int> orders(18, 0);

    orders[0] = (vertices[2] > vertices[3]);
    orders[1] = (vertices[1] > vertices[3]);
    orders[2] = (vertices[1] > vertices[2]);
    orders[3] = (vertices[0] > vertices[3]);
    orders[4] = (vertices[0] > vertices[2]);
    orders[5] = (vertices[0] > vertices[1]);

    std::tie(orders[6], orders[10]) = permuter.calculate_triangle_orders(
        vertices[1], vertices[2], vertices[3]);
    std::tie(orders[7], orders[11]) = permuter.calculate_triangle_orders(
        vertices[0], vertices[2], vertices[3]);
    std::tie(orders[8], orders[12]) = permuter.calculate_triangle_orders(
        vertices[0], vertices[1], vertices[3]);
    std::tie(orders[9], orders[13]) = permuter.calculate_triangle_orders(
        vertices[0], vertices[1], vertices[2]);

    auto tet_orders = permuter.calculate_tetrahedron_orders(
        vertices[0], vertices[1], vertices[2], vertices[3]);
    orders[14] = tet_orders[0];
    orders[15] = tet_orders[1];
    orders[16] = tet_orders[2];
    orders[17] = tet_orders[3];
    permuter.set_cell(cell_n, orders);
  }

  return permuter;
}
//-----------------------------------------------------------------------------
/// Make the DofMapPermuter that always returns the trivial permutation
/// @param[in] mesh The mesh
/// @param[in] element_dof_layout The layout of dofs in each cell
/// @return A DofMapPermuter for the mesh and dof layout
DofMapPermuter _empty_permutations(const mesh::Mesh mesh,
                                   const ElementDofLayout& element_dof_layout)
{
  DofMapPermuter permuter;
  permuter.set_dof_count(element_dof_layout.num_dofs());
  permuter.set_cell_count(mesh.num_entities(mesh.topology().dim()));
  return permuter;
}
//-----------------------------------------------------------------------------
DofMapPermuter
generate_cell_permutations(const mesh::Mesh mesh,
                           const ElementDofLayout& element_dof_layout)
{
  const mesh::CellType type = mesh.cell_type();
  switch (type)
  {
  case (mesh::CellType::triangle):
    return _generate_cell_permutations_triangle(mesh, element_dof_layout);
  case (mesh::CellType::tetrahedron):
    return _generate_cell_permutations_tetrahedron(mesh, element_dof_layout);
  default:
    LOG(WARNING) << "Dof permutations are not defined for this cell type. High "
                    "order elements may be incorrect.";
    return _empty_permutations(mesh, element_dof_layout);
  }
}
} // namespace fem
} // namespace dolfin
