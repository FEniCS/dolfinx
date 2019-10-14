// Copyright (C) 2019 Matthew Scroggs
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DofCellPermutations.h"
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshIterator.h>

namespace dolfin
{

namespace fem
{
//-----------------------------------------------------------------------------
DofMapPermuter::DofMapPermuter(const int dofs) : dof_count(dofs)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void DofMapPermuter::add_permutation(const std::vector<int> permutation, int order)
{
  _permutations.push_back(permutation);
  _permutation_orders.push_back(order);
}
//-----------------------------------------------------------------------------
void DofMapPermuter::set_cell_permutation(const int cell,const int permutation)
{
  _cell_permutation_numbers[cell] = permutation;
  _used[permutation] = true;
}
//-----------------------------------------------------------------------------
void DofMapPermuter::set_cell_permutation(const int cell, const std::vector<int> orders)
{
  set_cell_permutation(cell, get_permutation_number(orders));
}
//-----------------------------------------------------------------------------
void DofMapPermuter::prepare(const int cells)
{
  _total_options=1;
  for(int i=0;i<_permutation_orders.size();++i)
    _total_options *= _permutation_orders[i];
  _used.resize(_total_options,false);
  _cell_permutation_numbers.resize(cells,-1);
}
//-----------------------------------------------------------------------------
int DofMapPermuter::get_permutation_number(const std::vector<int> orders) const
{
  int out = 0;
  int base = 1;
  for(int i=0;i<orders.size();++i)
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
  for(int i=0;i<_permutation_orders.size();++i)
  {
    out[i] = (number / base) % _permutation_orders[i];
    base *= _permutation_orders[i];
  }
  return out;
}
//-----------------------------------------------------------------------------
void DofMapPermuter::generate_necessary_permutations()
{
  _cell_permutations.resize(_total_options,{});
  for(int p=0;p<_total_options;++p)
    if(_used[p])
    {
      std::vector<int> orders = get_orders(p);

      std::vector<int> permutation(dof_count);
      for(int i=0;i<dof_count;++i)
        permutation[i] = i;
      for (int i=0;i<orders.size();++i)
        for(int j=0;j<orders[i];++j)
          permutation = permute(permutation, _permutations[i]);

      _cell_permutations[p] = permutation;
    }
}
//-----------------------------------------------------------------------------
int DofMapPermuter::get_dof(const int cell, const int dof) const
{
  return _cell_permutations[_cell_permutation_numbers[cell]][dof];
}
//-----------------------------------------------------------------------------
const std::vector<int> DofMapPermuter::permute(std::vector<int> vec, std::vector<int> perm)
{
  std::vector<int> output(perm.size());
  for (int i=0;i<perm.size();++i)
    output[perm[i]] = vec[i];
  return output;
}
//-----------------------------------------------------------------------------
DofMapPermuter generate_cell_permutations(const mesh::Mesh mesh,
    const int vertex_dofs, const int edge_dofs, const int face_dofs, const int volume_dofs)
{
  const mesh::CellType type = mesh.cell_type();
  switch (type)
  {
    case (mesh::CellType::quadrilateral):
      return generate_cell_permutations_quadrilateral(mesh, vertex_dofs, edge_dofs, face_dofs);
    case (mesh::CellType::triangle):
      return generate_cell_permutations_triangle(mesh, vertex_dofs, edge_dofs, face_dofs);
    default:
      throw std::runtime_error("Dof ordering on this cell type is not implemented.");
  }
}
//-----------------------------------------------------------------------------
DofMapPermuter generate_cell_permutations_triangle(const mesh::Mesh mesh,
    const int vertex_dofs, const int edge_dofs, const int face_dofs)
{
  const int dof_count = 3*vertex_dofs + 3*edge_dofs + face_dofs;
  DofMapPermuter output(dof_count);

  float root = std::sqrt(8*face_dofs+1);
  assert(root == floor(root) && root%2 == 1);
  int side_length = (root-1)/2; // side length of the triangle of face dofs

  // Make edge flipping permutations
  for (int edge=0;edge<3;++edge)
  {
    std::vector<int> flip(dof_count);
    int j=0;
    for(int dof=0;dof<3*vertex_dofs+edge*edge_dofs;++dof)
      flip[j++] = dof;
    for(int dof=3*vertex_dofs+(edge+1)*edge_dofs-1;dof>=3*vertex_dofs+edge*edge_dofs;--dof)
      flip[j++] = dof;
    for(int dof=3*vertex_dofs+(edge+1)*edge_dofs;dof<3*vertex_dofs+3*edge_dofs+face_dofs;++dof)
      flip[j++] = dof;
    assert(j == dof_count);
    output.add_permutation(flip, 2);
  }

  // Make permutation that rotates the face dofs
  {
    std::vector<int> rotation(dof_count);
    int j=0;
    for(int dof=0;dof<3*vertex_dofs+3*edge_dofs;++dof)
      rotation[j++] = dof;
    // face
    for(int st=side_length-1;st>=0;--st){
      int dof = 3*vertex_dofs + 3*edge_dofs + st;
      for(int add=side_length-1;add>=side_length-st-1;--add)
        rotation[j++] = (dof += add);
    }
    assert(j == dof_count);
    output.add_permutation(rotation,3);
  }

  // Make permutation that reflects the face dofs
  {
    std::vector<int> reflection(dof_count);
    int j=0;
    for(int dof=0;dof<3*vertex_dofs+3*edge_dofs;++dof)
      reflection[j++] = dof;
    // face
    for(int st=0;st<side_length;++st){
      int dof = 3*vertex_dofs + 3*edge_dofs + st;
      for(int add=side_length+1;add>=st+2;dof+= (add--))
        reflection[j++] = dof;
    }
    assert(j == dof_count);
    output.add_permutation(reflection, 2);
  }

  int cells = mesh.num_entities(mesh.topology().dim());
  output.prepare(cells);

  for(int cell_n=0;cell_n<cells;++cell_n){
    const mesh::MeshEntity cell(mesh, 2, cell_n);
    const std::int32_t* vertices = cell.entities(0);
    std::vector<int> orders(5);
    orders[0] = (vertices[1] > vertices[2]);
    orders[1] = (vertices[0] > vertices[2]);
    orders[2] = (vertices[0] > vertices[1]);

    if(vertices[0] < vertices[1] && vertices[0] < vertices[2])
    {
      orders[3] = 0;
      orders[4] = (vertices[1] > vertices[2]);
    }
    if(vertices[1] < vertices[0] && vertices[1] < vertices[2])
    {
      orders[3] = 1;
      orders[4] = (vertices[2] > vertices[0]);
    }
    if(vertices[2] < vertices[0] && vertices[2] < vertices[1])
    {
      orders[3] = 2;
      orders[4] = (vertices[0] > vertices[1]);
    }
    output.set_cell_permutation(cell_n, orders);
  }

  output.generate_necessary_permutations();

  return output;
}
//-----------------------------------------------------------------------------
DofMapPermuter generate_cell_permutations_quadrilateral(const mesh::Mesh mesh,
    const int vertex_dofs, const int edge_dofs, const int face_dofs)
{
  const int dof_count = 4*vertex_dofs + 4*edge_dofs + face_dofs;
  DofMapPermuter output(dof_count);
  return output;
}
} // namespace fem
} // namespace dolfin
