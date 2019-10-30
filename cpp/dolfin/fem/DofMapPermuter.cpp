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
int get_num_permutations(mesh::CellType cell_type)
{
  switch (cell_type)
  {
  case (mesh::CellType::triangle):
    return 5;
  case (mesh::CellType::tetrahedron):
    return 18;
  default:
    LOG(WARNING) << "Dof permutations are not defined for this cell type. High "
                    "order elements may be incorrect.";
    return 0;
  }
}
//-----------------------------------------------------------------------------
/// Calculates the permutation orders for a triangle
/// @param[in] v1 The global vertex number of the triangle's first vertex
/// @param[in] v2 The global vertex number of the triangle's second vertex
/// @param[in] v3 The global vertex number of the triangle's third vertex
/// @return The rotation and reflection orders for the triangle
std::array<int, 2> calculate_triangle_orders(int v1, int v2, int v3)
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
std::array<int, 4> calculate_tetrahedron_orders(int v1, int v2, int v3, int v4)
{
  if (v1 < v2 && v1 < v3 && v1 < v4)
  {
    const std::array<int, 2> tri_orders = calculate_triangle_orders(v2, v3, v4);
    return {0, 0, tri_orders[0], tri_orders[1]};
  }
  if (v2 < v1 && v2 < v3 && v2 < v4)
  {
    const std::array<int, 2> tri_orders = calculate_triangle_orders(v3, v1, v4);
    return {1, 0, tri_orders[0], tri_orders[1]};
  }
  if (v3 < v1 && v3 < v2 && v3 < v4)
  {
    const std::array<int, 2> tri_orders = calculate_triangle_orders(v1, v2, v4);
    return {2, 0, tri_orders[0], tri_orders[1]};
  }
  if (v4 < v1 && v4 < v2 && v4 < v3)
  {
    const std::array<int, 2> tri_orders = calculate_triangle_orders(v2, v1, v3);
    return {0, 1, tri_orders[0], tri_orders[1]};
  }
  throw std::runtime_error(
      "Two of a tetrahedron's vertices appear to be equal.");
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
/// Makes permutations to rotate and reflect the dofs in a tetrahedron
/// @param[in] edge_dofs Number of dofs on the interior of the tetrahedron
/// @return Permutations to rotate and reflect the dofs
std::array<std::vector<int>, 4>
tetrahedron_rotations_and_reflection(const int volume_dofs, const int blocksize)
{
  // FIXME: This function assumes the layout of the dofs are in a triangle
  // shape. This is true for Lagrange space, but not true for eg N1curl
  // spaces. This will only be called at most once for each mesh
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
    auto base_faces = triangle_rotation_and_reflection(face_dofs, 1);

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
void apply_permutation(
    Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic>& permutations,
    const int index, const Eigen::Array<int, Eigen::Dynamic, 1>& dofs,
    const std::vector<int>& base_permutation, int order)
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
DofMapPermuter::DofMapPermuter(const mesh::Mesh& mesh,
                               const ElementDofLayout& element_dof_layout)
{
  const int num_cells = mesh.num_entities(mesh.topology().dim());
  const int num_permutations = get_num_permutations(mesh.cell_type());
  _cell_orders = Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic>::Zero(
      num_cells, num_permutations);
  _permutations = generate_recursive(mesh, element_dof_layout);

  switch (mesh.cell_type())
  {
  case (mesh::CellType::triangle):
    _cell_orders = set_orders_triangle(mesh, element_dof_layout);
    break;
  case (mesh::CellType::tetrahedron):
    _cell_orders = set_orders_tetrahedron(mesh, element_dof_layout);
    break;
  default:
    throw std::runtime_error(
        "Unrecognised cell type."); // The function should exit before this is
                                    // reached
  }
}
//-----------------------------------------------------------------------------
std::vector<int> DofMapPermuter::cell_permutation(const int cell) const
{
  std::vector<int> p(_permutations.cols());
  std::iota(p.begin(), p.end(), 0);

  for (int i = 0; i < _cell_orders.cols(); ++i)
  {
    for (int j = 0; j < _cell_orders(cell, i); ++j)
    {
      std::vector<int> temp(p);
      for (std::size_t k = 0; k < p.size(); ++k)
        p[_permutations(i, k)] = temp[k];
    }
  }

  return p;
}
//-----------------------------------------------------------------------------
// private:
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic>
DofMapPermuter::generate_recursive(const mesh::Mesh& mesh,
                                   const ElementDofLayout& element_dof_layout)
{
  if (element_dof_layout.num_sub_dofmaps() == 0)
  {
    switch (mesh.cell_type())
    {
    case (mesh::CellType::triangle):
      return generate_triangle(mesh, element_dof_layout);
    case (mesh::CellType::tetrahedron):
      return generate_tetrahedron(mesh, element_dof_layout);
    default:
      throw std::runtime_error(
          "Unrecognised cell type."); // The function should exit before this is
                                      // reached
    }
  }

  const int num_permutations = get_num_permutations(mesh.cell_type());
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic> output(
      num_permutations, element_dof_layout.num_dofs());

  for (int i = 0; i < element_dof_layout.num_sub_dofmaps(); ++i)
  {
    auto sub_view = element_dof_layout.sub_view({i});
    const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic> sub_perm
        = generate_recursive(mesh, *element_dof_layout.sub_dofmap({i}));
    for (int p = 0; p < num_permutations; ++p)
      for (std::size_t j = 0; j < sub_view.size(); ++j)
        output(p, sub_view[j]) = sub_view[sub_perm(p, j)];
  }
  return output;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic>
DofMapPermuter::generate_triangle(const mesh::Mesh& mesh,
                                  const ElementDofLayout& element_dof_layout)
{
  const int num_permutations = get_num_permutations(mesh.cell_type());
  const int dof_count = element_dof_layout.num_dofs();

  const int edge_dofs = element_dof_layout.num_entity_dofs(1);
  const int face_dofs = element_dof_layout.num_entity_dofs(2);
  const int edge_bs = element_dof_layout.entity_block_size(1);
  const int face_bs = element_dof_layout.entity_block_size(2);

  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic> permutations(
      num_permutations, dof_count);
  for (int i = 0; i < permutations.cols(); ++i)
    permutations.col(i) = i;

  // Make edge flipping permutations
  std::vector<int> base_flip = edge_flip(edge_dofs, edge_bs);
  for (int edge_n = 0; edge_n < 3; ++edge_n)
  {
    auto edge = element_dof_layout.entity_dofs(1, edge_n);
    apply_permutation(permutations, edge_n, edge, base_flip, 2);
  }
  // Make permutations that rotate and reflect the face dofs
  auto base_faces = triangle_rotation_and_reflection(face_dofs, face_bs);
  auto face = element_dof_layout.entity_dofs(2, 0);
  apply_permutation(permutations, 3, face, base_faces[0], 3);
  apply_permutation(permutations, 4, face, base_faces[1], 2);

  return permutations;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic>
DofMapPermuter::generate_tetrahedron(const mesh::Mesh& mesh,
                                     const ElementDofLayout& element_dof_layout)
{
  const int num_permutations = get_num_permutations(mesh.cell_type());
  const int dof_count = element_dof_layout.num_dofs();

  const int edge_dofs = element_dof_layout.num_entity_dofs(1);
  const int face_dofs = element_dof_layout.num_entity_dofs(2);
  const int volume_dofs = element_dof_layout.num_entity_dofs(3);
  const int edge_bs = element_dof_layout.entity_block_size(1);
  const int face_bs = element_dof_layout.entity_block_size(2);
  const int volume_bs = element_dof_layout.entity_block_size(3);

  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic> permutations(
      num_permutations, dof_count);
  for (int i = 0; i < permutations.cols(); ++i)
    permutations.col(i) = i;

  // Make edge flipping permutations
  std::vector<int> base_flip = edge_flip(edge_dofs, edge_bs);
  for (int edge_n = 0; edge_n < 6; ++edge_n)
  {
    auto edge = element_dof_layout.entity_dofs(1, edge_n);
    apply_permutation(permutations, edge_n, edge, base_flip, 2);
  }
  // Make permutations that rotate and reflect the face dofs
  auto base_faces = triangle_rotation_and_reflection(face_dofs, face_bs);
  for (int face_n = 0; face_n < 4; ++face_n)
  {
    auto face = element_dof_layout.entity_dofs(2, face_n);
    apply_permutation(permutations, 6 + 2 * face_n, face, base_faces[0], 3);
    apply_permutation(permutations, 6 + 2 * face_n + 1, face, base_faces[1], 2);
  }
  // Make permutations that rotate and reflect the interior dofs
  auto base_interiors
      = tetrahedron_rotations_and_reflection(volume_dofs, volume_bs);
  auto interior = element_dof_layout.entity_dofs(3, 0);
  apply_permutation(permutations, 14, interior, base_interiors[0], 3);
  apply_permutation(permutations, 15, interior, base_interiors[1], 3);
  apply_permutation(permutations, 16, interior, base_interiors[2], 3);
  apply_permutation(permutations, 17, interior, base_interiors[3], 2);
  return permutations;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic>
DofMapPermuter::set_orders_triangle(const mesh::Mesh& mesh,
                                    const ElementDofLayout& element_dof_layout)
{
  const int num_cells = mesh.num_entities(mesh.topology().dim());
  const int num_permutations = get_num_permutations(mesh.cell_type());
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic> cell_orders(
      num_cells, num_permutations);

  // Set orders for each cell
  for (int cell_n = 0; cell_n < num_cells; ++cell_n)
  {
    const mesh::MeshEntity cell(mesh, 2, cell_n);
    const std::int32_t* vertices = cell.entities(0);
    cell_orders(cell_n, 0) = (vertices[1] > vertices[2]);
    cell_orders(cell_n, 1) = (vertices[0] > vertices[2]);
    cell_orders(cell_n, 2) = (vertices[0] > vertices[1]);

    auto tri_orders
        = calculate_triangle_orders(vertices[0], vertices[1], vertices[2]);
    cell_orders(cell_n, 3) = tri_orders[0];
    cell_orders(cell_n, 4) = tri_orders[0];
  }

  return cell_orders;
}
//-----------------------------------------------------------------------------
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic>
DofMapPermuter::set_orders_tetrahedron(
    const mesh::Mesh& mesh, const ElementDofLayout& element_dof_layout)
{
  const int num_cells = mesh.num_entities(mesh.topology().dim());
  const int num_permutations = get_num_permutations(mesh.cell_type());
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic> cell_orders(
      num_cells, num_permutations);

  // Set orders for each cell
  for (int cell_n = 0; cell_n < num_cells; ++cell_n)
  {
    const mesh::MeshEntity cell(mesh, 3, cell_n);
    const std::int32_t* vertices = cell.entities(0);

    cell_orders(cell_n, 0) = (vertices[2] > vertices[3]);
    cell_orders(cell_n, 1) = (vertices[1] > vertices[3]);
    cell_orders(cell_n, 2) = (vertices[1] > vertices[2]);
    cell_orders(cell_n, 3) = (vertices[0] > vertices[3]);
    cell_orders(cell_n, 4) = (vertices[0] > vertices[2]);
    cell_orders(cell_n, 5) = (vertices[0] > vertices[1]);

    std::array<int, 2> tri_orders
        = calculate_triangle_orders(vertices[1], vertices[2], vertices[3]);
    cell_orders(cell_n, 6) = tri_orders[0];
    cell_orders(cell_n, 7) = tri_orders[1];
    tri_orders
        = calculate_triangle_orders(vertices[0], vertices[2], vertices[3]);
    cell_orders(cell_n, 8) = tri_orders[0];
    cell_orders(cell_n, 9) = tri_orders[1];
    tri_orders
        = calculate_triangle_orders(vertices[0], vertices[1], vertices[3]);
    cell_orders(cell_n, 10) = tri_orders[0];
    cell_orders(cell_n, 11) = tri_orders[1];
    tri_orders
        = calculate_triangle_orders(vertices[0], vertices[1], vertices[2]);
    cell_orders(cell_n, 12) = tri_orders[0];
    cell_orders(cell_n, 13) = tri_orders[1];

    std::array<int, 4> tet_orders = calculate_tetrahedron_orders(
        vertices[0], vertices[1], vertices[2], vertices[3]);
    cell_orders(cell_n, 14) = tet_orders[0];
    cell_orders(cell_n, 15) = tet_orders[1];
    cell_orders(cell_n, 16) = tet_orders[2];
    cell_orders(cell_n, 17) = tet_orders[3];
  }

  return cell_orders;
}
//-----------------------------------------------------------------------------
} // namespace fem
} // namespace dolfin
