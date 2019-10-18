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
void DofMapPermuter::set_dof_count(const int dofs) { dof_count = dofs; }
//-----------------------------------------------------------------------------
void DofMapPermuter::add_permutation(const std::vector<int> permutation, int order)
{
  _permutations.push_back(permutation);
  _permutation_orders.push_back(order);
}
//-----------------------------------------------------------------------------
void DofMapPermuter::set_cell(const int cell,const int permutation)
{
  set_cell(cell, get_orders(permutation));
}
//-----------------------------------------------------------------------------
void DofMapPermuter::set_cell(const int cell, const std::vector<int> orders)
{
  _cell_orders[cell] = orders;
}
//-----------------------------------------------------------------------------
void DofMapPermuter::set_cell_count(const int cells)
{
  _cell_orders.resize(cells,{0,0,0,0});
}
//-----------------------------------------------------------------------------
int DofMapPermuter::get_permutation_number(const std::vector<int> orders) const
{
  int out = 0;
  int base = 1;
  for (int i = 0; i < orders.size(); ++i)
  {
    out += base * (orders[i] % _permutation_orders[i]);
    base *= _permutation_orders[i];
  }
  return out;
}
//-----------------------------------------------------------------------------
std::vector<int> DofMapPermuter::get_orders(const int number) const
{
  std::vector<int> out(_permutation_orders.size());
  int base = 1;
  for (int i = 0; i < _permutation_orders.size(); ++i)
  {
    out[i] = (number / base) % _permutation_orders[i];
    base *= _permutation_orders[i];
  }
  return out;
}
//-----------------------------------------------------------------------------
std::vector<int> DofMapPermuter::permute(std::vector<int> vec, std::vector<int> perm) const
{
  std::vector<int> output(perm.size());
  for (int i = 0; i < perm.size(); ++i)
    output[perm[i]] = vec[i];
  return output;
}
//-----------------------------------------------------------------------------
std::vector<int> DofMapPermuter::cell_permutation(const int cell) const
{
  std::vector<int> orders = _cell_orders[cell];

  std::vector<int> permutation(dof_count);
  for (int i = 0; i < dof_count; ++i)
    permutation[i] = i;

  for (int i = 0; i < orders.size(); ++i)
    for (int j = 0; j < orders[i]; ++j)
      permutation = permute(permutation, _permutations[i]);

  return permutation;
}
//-----------------------------------------------------------------------------
DofMapPermuter generate_cell_permutations(const mesh::Mesh mesh,
    const int vertex_dofs, const int edge_dofs, const int face_dofs, const int volume_dofs)
{
  const mesh::CellType type = mesh.cell_type();
  switch (type)
  {
  case (mesh::CellType::triangle):
    return generate_cell_permutations_triangle(mesh, vertex_dofs, edge_dofs,
                                               face_dofs);
  case (mesh::CellType::tetrahedron):
    return generate_cell_permutations_tetrahedron(mesh, vertex_dofs, edge_dofs,
                                                  face_dofs, volume_dofs);
  default:
    LOG(WARNING) << "Dof permutations are not defined for this cell type. High "
                    "order elements may be incorrect.";
    return empty_permutations(mesh, vertex_dofs, edge_dofs, face_dofs,
                              volume_dofs);
  }
}
//-----------------------------------------------------------------------------
std::vector<int> edge_flip(const int edge_dofs)
{
  std::vector<int> flip(edge_dofs);
  for (int i = 0; i < edge_dofs; ++i)
    flip[i] = edge_dofs - 1 - i;
  return flip;
}
//-----------------------------------------------------------------------------
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
    assert(j == dof_count);
  }

  return std::make_pair(rotation, reflection);
}
//-----------------------------------------------------------------------------
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
    for (int j = 0; j < face.size(); ++j)
    {
      rotation1[face[j]] = face[base_face_rotation[j]];
      rotation2[face2[j]] = face2[base_face_rotation[j]];
      reflection[face[j]] = face[base_face_reflection[j]];
    }
  }

  return std::make_tuple(rotation1, rotation2, reflection);
}
//-----------------------------------------------------------------------------
std::pair<int, int> calculate_triangle_orders(int v1, int v2, int v3)
{
  if (v1 < v2 && v1 < v3)
    return std::make_pair(0, v2 > v3);
  if (v2 < v1 && v2 < v3)
    return std::make_pair(1, v3 > v1);
  if (v3 < v1 && v3 < v2)
    return std::make_pair(2, v1 > v2);
}
//-----------------------------------------------------------------------------
std::tuple<int, int, int, int> calculate_tetrahedron_orders(int v1, int v2,
                                                            int v3, int v4)
{
  if (v1 < v2 && v1 < v3 && v1 < v4)
  {
    int a;
    int b;
    std::tie(a, b) = calculate_triangle_orders(v2, v3, v4);
    return std::make_tuple(0, 0, a, b);
  }
  if (v2 < v1 && v2 < v3 && v2 < v4)
  {
    int a;
    int b;
    std::tie(a, b) = calculate_triangle_orders(v3, v1, v4);
    return std::make_tuple(1, 0, a, b);
  }
  if (v3 < v1 && v3 < v2 && v3 < v4)
  {
    int a;
    int b;
    std::tie(a, b) = calculate_triangle_orders(v1, v2, v4);
    return std::make_tuple(2, 0, a, b);
  }
  if (v4 < v1 && v4 < v2 && v4 < v3)
  {
    int a;
    int b;
    std::tie(a, b) = calculate_triangle_orders(v2, v1, v3);
    return std::make_tuple(0, 1, a, b);
  }
}
//-----------------------------------------------------------------------------
DofMapPermuter generate_cell_permutations_triangle(const mesh::Mesh mesh,
    const int vertex_dofs, const int edge_dofs, const int face_dofs)
{
  // FIXME: This function assumes VTK-like ordering of the dofs. It should use
  // ElementDofLayout instead.
  const int dof_count = 3*vertex_dofs + 3*edge_dofs + face_dofs;
  DofMapPermuter output;
  output.set_dof_count(dof_count);

  float root = std::sqrt(8*face_dofs+1);
  assert(root == floor(root) && root%2 == 1);
  int side_length = (root-1)/2; // side length of the triangle of face dofs

  // Make edge flipping permutations
  std::vector<int> base_flip = edge_flip(edge_dofs);
  for (int edge_n = 0; edge_n < 3; ++edge_n)
  {
    std::vector<int> flip(dof_count);
    std::iota(flip.begin(), flip.end(), 0);
    // FIXME: infer the following from ElementDofLayout
    std::vector<int> edge(edge_dofs);
    std::iota(edge.begin(), edge.end(), 3 * vertex_dofs + edge_n * edge_dofs);
    for (int j = 0; j < edge.size(); ++j)
      flip[edge[j]] = edge[base_flip[j]];
    output.add_permutation(flip, 2);
  }

  std::vector<int> base_face_rotation;
  std::vector<int> base_face_reflection;
  tie(base_face_rotation, base_face_reflection)
      = triangle_rotation_and_reflection(face_dofs);
  // Make permutation that rotates the face dofs
  {
    std::vector<int> rotation(dof_count);
    std::iota(rotation.begin(), rotation.end(), 0);
    // FIXME: infer the following from ElementDofLayout
    std::vector<int> face(face_dofs);
    std::iota(face.begin(), face.end(), 3 * vertex_dofs + 3 * edge_dofs);
    for (int j = 0; j < face.size(); ++j)
      rotation[face[j]] = face[base_face_rotation[j]];
    output.add_permutation(rotation,3);
  }

  // Make permutation that reflects the face dofs
  {
    std::vector<int> reflection(dof_count);
    std::iota(reflection.begin(), reflection.end(), 0);
    // FIXME: infer the following from ElementDofLayout
    std::vector<int> face(face_dofs);
    std::iota(face.begin(), face.end(), 3 * vertex_dofs + 3 * edge_dofs);
    for (int j = 0; j < face.size(); ++j)
      reflection[face[j]] = face[base_face_reflection[j]];
    output.add_permutation(reflection, 2);
  }

  int cells = mesh.num_entities(mesh.topology().dim());
  output.set_cell_count(cells);

  for (int cell_n = 0; cell_n < cells; ++cell_n)
  {
    const mesh::MeshEntity cell(mesh, 2, cell_n);
    const std::int32_t* vertices = cell.entities(0);
    std::vector<int> orders(5);
    orders[0] = (vertices[1] > vertices[2]);
    orders[1] = (vertices[0] > vertices[2]);
    orders[2] = (vertices[0] > vertices[1]);

    std::tie(orders[3], orders[4])
        = calculate_triangle_orders(vertices[0], vertices[1], vertices[2]);

    output.set_cell(cell_n, orders);
  }

  return output;
}
//-----------------------------------------------------------------------------
DofMapPermuter generate_cell_permutations_tetrahedron(const mesh::Mesh mesh,
                                                      const int vertex_dofs,
                                                      const int edge_dofs,
                                                      const int face_dofs,
                                                      const int volume_dofs)
{
  // FIXME: This function assumes VTK-like ordering of the dofs. It should use
  // ElementDofLayout instead.
  const int dof_count
      = 4 * vertex_dofs + 6 * edge_dofs + 4 * face_dofs + volume_dofs;
  DofMapPermuter output;
  output.set_dof_count(dof_count);

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
    // FIXME: infer the following from ElementDofLayout
    std::vector<int> edge(edge_dofs);
    std::iota(edge.begin(), edge.end(), 4 * vertex_dofs + edge_n * edge_dofs);
    for (int j = 0; j < edge.size(); ++j)
      flip[edge[j]] = edge[base_flip[j]];
    output.add_permutation(flip, 2);
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
    // FIXME: infer the following from ElementDofLayout
    std::vector<int> face(face_dofs);
    std::iota(face.begin(), face.end(),
              4 * vertex_dofs + 6 * edge_dofs + face_n * face_dofs);
    for (int j = 0; j < face.size(); ++j)
      rotation[face[j]] = face[base_face_rotation[j]];
    output.add_permutation(rotation, 3);
  }

  // Make permutations that reflect the face dofs
  for (int face_n = 0; face_n < 4; ++face_n)
  {
    std::vector<int> reflection(dof_count);
    std::iota(reflection.begin(), reflection.end(), 0);
    // FIXME: infer the following from ElementDofLayout
    std::vector<int> face(face_dofs);
    std::iota(face.begin(), face.end(),
              4 * vertex_dofs + 6 * edge_dofs + face_n * face_dofs);
    for (int j = 0; j < face.size(); ++j)
      reflection[face[j]] = face[base_face_reflection[j]];
    output.add_permutation(reflection, 2);
  }

  std::vector<int> base_interior_rotation1;
  std::vector<int> base_interior_rotation2;
  std::vector<int> base_interior_reflection;
  tie(base_interior_rotation1, base_interior_rotation2,
      base_interior_reflection)
      = tetrahedron_rotations_and_reflection(volume_dofs);
  // FIXME: infer the following from ElementDofLayout
  std::vector<int> interior(volume_dofs);
  std::iota(interior.begin(), interior.end(),
            4 * vertex_dofs + 6 * edge_dofs + 4 * face_dofs);
  {
    std::vector<int> rotation(dof_count);
    std::iota(rotation.begin(), rotation.end(), 0);
    for (int j = 0; j < interior.size(); ++j)
      rotation[interior[j]] = interior[base_interior_rotation1[j]];
    output.add_permutation(rotation, 3);
  }
  {
    std::vector<int> rotation(dof_count);
    std::iota(rotation.begin(), rotation.end(), 0);
    for (int j = 0; j < interior.size(); ++j)
      rotation[interior[j]] = interior[base_interior_rotation2[j]];
    output.add_permutation(rotation, 3);
  }
  {
    std::vector<int> rotation(dof_count);
    std::iota(rotation.begin(), rotation.end(), 0);
    for (int j = 0; j < interior.size(); ++j)
      rotation[interior[j]]
          = interior[base_interior_rotation2
                         [base_interior_rotation2[base_interior_rotation1[j]]]];
    output.add_permutation(rotation, 3);
  }
  {
    std::vector<int> reflection(dof_count);
    std::iota(reflection.begin(), reflection.end(), 0);
    for (int j = 0; j < interior.size(); ++j)
      reflection[interior[j]] = interior[base_interior_reflection[j]];
    output.add_permutation(reflection, 2);
  }

  int cells = mesh.num_entities(mesh.topology().dim());
  output.set_cell_count(cells);

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

    std::tie(orders[6], orders[10])
        = calculate_triangle_orders(vertices[1], vertices[2], vertices[3]);
    std::tie(orders[7], orders[11])
        = calculate_triangle_orders(vertices[0], vertices[2], vertices[3]);
    std::tie(orders[8], orders[12])
        = calculate_triangle_orders(vertices[0], vertices[1], vertices[3]);
    std::tie(orders[9], orders[13])
        = calculate_triangle_orders(vertices[0], vertices[1], vertices[2]);

    std::tie(orders[14], orders[15], orders[16], orders[17])
        = calculate_tetrahedron_orders(vertices[0], vertices[1], vertices[2],
                                       vertices[3]);
    output.set_cell(cell_n, orders);
  }

  return output;
}
//-----------------------------------------------------------------------------
DofMapPermuter empty_permutations(const mesh::Mesh mesh, const int vertex_dofs,
                                  const int edge_dofs, const int face_dofs,
                                  const int volume_dofs)
{
  // This function returns a permuter that contains only empty permutations
  DofMapPermuter output;
  const mesh::CellType type = mesh.cell_type();
  int dof_count = 0;
  dof_count += vertex_dofs * mesh::cell_num_entities(type, 0);
  dof_count += edge_dofs * mesh::cell_num_entities(type, 1);
  dof_count += face_dofs * mesh::cell_num_entities(type, 2);
  dof_count += volume_dofs * mesh::cell_num_entities(type, 3);

  output.set_dof_count(dof_count);

  int cells = mesh.num_entities(mesh.topology().dim());
  output.set_cell_count(cells);
  for (int cell_n = 0; cell_n < cells; ++cell_n)
    output.set_cell(cell_n, {});

  return output;
}
} // namespace fem
} // namespace dolfin
