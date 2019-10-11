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
void DofMapPermuter::add_edge_flip(const std::vector<int> flip)
{
  _edge_flips.push_back(flip);
}
//-----------------------------------------------------------------------------
void DofMapPermuter::set_reflection(const std::vector<int> reflection)
{
  _reflection=reflection;
}
//-----------------------------------------------------------------------------
void DofMapPermuter::set_rotation(const std::vector<int> rotation, int order)
{
  _rotation=rotation;
  _rotation_order=order;
}
//-----------------------------------------------------------------------------
void DofMapPermuter::set_cell_permutation(const int cell,const int permutation)
{
  _permutations_of_cells[cell]=permutation;
  _used[permutation] = true;
}
//-----------------------------------------------------------------------------
void DofMapPermuter::set_cell_permutation(const int cell,const int rotations, const int reflections, const std::vector<int> edge_flips)
{
  set_cell_permutation(cell, get_permutation_number(rotations, reflections, edge_flips));
}
//-----------------------------------------------------------------------------
void DofMapPermuter::prepare(const int cells)
{
  _total_options = _rotation_order * 2 * (1 << _edge_flips.size());
  _used.resize(_total_options,false);
  _permutations_of_cells.resize(cells,-1);
}
//-----------------------------------------------------------------------------
const int DofMapPermuter::get_permutation_number(const int rotations, const int reflections, const std::vector<int> edge_flips)
{
  int out = reflections + 2*rotations;
  for(int i=0;i<edge_flips.size();++i)
    out += 2 * _rotation_order * (1 << edge_flips[i]);
  return out;
}
//-----------------------------------------------------------------------------
void DofMapPermuter::generate_necessary_permutations()
{
  _permutations.resize(_total_options,{});
  for(int p=0;p<_total_options;++p)
    if(_used[p])
    {
      int rotations = p % _rotation_order;
      int reflections = (p/_rotation_order) % 2;
      std::vector<int> edge_flips;
      int left = p/(_rotation_order*2);
      for(int j=0;left>0;++j)
      {
        if(left%2==1)
          edge_flips.push_back(j);
        left /= 2;
      }

      std::cout << p << " -> " << rotations << " " << reflections << std::endl;

      std::vector<int> permutation(dof_count);
      for(int i=0;i<dof_count;++i)
        permutation[i] = i;
      for (int i=0;i<rotations;++i)
        permutation = permute(permutation, _rotation);
      if (reflections>0)
        permutation = permute(permutation, _reflection);
      for(int i=0;i<edge_flips.size();++i)
        permutation = permute(permutation, _edge_flips[i]);

      _permutations[p] = permutation;
    }
}
//-----------------------------------------------------------------------------
int DofMapPermuter::get_dof(const int cell, const int dof) const
{
  return _permutations[_permutations_of_cells[cell]][dof];
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

  // Make permutation that rotates the triangle
  {
    std::vector<int> rotation(dof_count);
    int j=0;
    // vertices
    for(int dof=vertex_dofs;dof<2*vertex_dofs;++dof)
      rotation[j++] = dof;
    for(int dof=2*vertex_dofs;dof<3*vertex_dofs;++dof)
      rotation[j++] = dof;
    for(int dof=0;dof<vertex_dofs;++dof)
      rotation[j++] = dof;
    // edges
    for(int dof=3*vertex_dofs+2*edge_dofs-1;dof>=3*vertex_dofs+edge_dofs;--dof)
      rotation[j++] = dof;
    for(int dof=3*vertex_dofs+3*edge_dofs-1;dof>=3*vertex_dofs+2*edge_dofs;--dof)
      rotation[j++] = dof;
    for(int dof=3*vertex_dofs;dof<3*vertex_dofs+edge_dofs;++dof)
      rotation[j++] = dof;
    // face
    for(int st=side_length-1;st>=0;--st){
      int dof = 3*vertex_dofs + 3*edge_dofs + st;
      for(int add=side_length-1;add>=side_length-st-1;--add)
        rotation[j++] = (dof += add);
    }
    assert(j == dof_count);
    output.set_rotation(rotation,3);
  }

  // Make permutation that reflects the triangle
  {
    std::vector<int> reflection(dof_count);
    int j=0;
    // vertices
    for(int dof=0;dof<vertex_dofs;++dof)
      reflection[j++] = dof;
    for(int dof=2*vertex_dofs;dof<3*vertex_dofs;++dof)
      reflection[j++] = dof;
    for(int dof=vertex_dofs;dof<2*vertex_dofs;++dof)
      reflection[j++] = dof;
    // edges
    for(int dof=3*vertex_dofs+edge_dofs-1;dof>=3*vertex_dofs;--dof)
      reflection[j++] = dof;
    for(int dof=3*vertex_dofs+2*edge_dofs;dof<3*vertex_dofs+3*edge_dofs;++dof)
      reflection[j++] = dof;
    for(int dof=3*vertex_dofs+edge_dofs;dof<3*vertex_dofs+2*edge_dofs;++dof)
      reflection[j++] = dof;
    // face
    for(int st=0;st<side_length;++st){
      int dof = 3*vertex_dofs + 3*edge_dofs + st;
      for(int add=side_length+1;add>=st+2;--add)
        reflection[j++] = (dof += add);
    }
    assert(j == dof_count);
    output.set_reflection(reflection);
  }

  int cells = 2; // TODO: number of cells
  output.prepare(cells);

  for(int i=0;i<cells;++i){
    const mesh::MeshEntity cell(mesh, 2, i);
    const std::int32_t* vertices = cell.entities(0);
    int rotations;
    int reflections;
    if(vertices[0] < vertices[1] && vertices[0] < vertices[2])
    {
      rotations = 0;
      reflections = (vertices[1] > vertices[2]);
    }
    if(vertices[1] < vertices[0] && vertices[1] < vertices[2])
    {
      rotations = 1;
      reflections = (vertices[2] > vertices[0]);
    }
    if(vertices[2] < vertices[0] && vertices[2] < vertices[1])
    {
      rotations = 2;
      reflections = (vertices[0] > vertices[1]);
    }
    std::cout << "perm:: " << i << " " << rotations << " " << reflections << " :: ";
    for(int j=0;j<3;++j) std::cout << vertices[j] << ",";
    std::cout << std::endl;
    output.set_cell_permutation(i, rotations, reflections);
  }

  output.generate_necessary_permutations();

  // 
  /* int mpi_rank = dolfin::MPI::rank(mpi_comm);
  std::vector<std::int64_t> local_to_global;
  std::array<int, 4> num_vertices_local;

  // Classify all nodes
  std::set<std::int64_t> local_vertices;
  std::set<std::int64_t> shared_vertices;
  std::set<std::int64_t> ghost_vertices;
  std::set<std::int64_t> non_vertex_nodes;

  const std::int32_t num_cells = cell_nodes.rows();
  const std::int32_t num_nodes_per_cell = cell_nodes.cols();
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    // Loop over vertex nodes
    for (std::int32_t v = 0; v < num_vertices_per_cell; ++v)
    {
      // Get global node index
      std::int64_t q = cell_nodes(c, v);
      auto shared_it = point_to_procs.find(q);
      if (shared_it == point_to_procs.end())
        local_vertices.insert(q);
      else
      {
        // If lowest ranked sharing process is greather than this process,
        // then it is owner
        if (*(shared_it->second.begin()) > mpi_rank)
          shared_vertices.insert(q);
        else
          ghost_vertices.insert(q);
      }
    }
    // Non-vertex nodes
    for (std::int32_t v = num_vertices_per_cell; v < num_nodes_per_cell; ++v)
    {
      // Get global node index
      std::int64_t q = cell_nodes(c, v);
      non_vertex_nodes.insert(q);
    }
  }
  */
  return output;
}
//-----------------------------------------------------------------------------
DofMapPermuter generate_cell_permutations_quadrilateral(const mesh::Mesh mesh,
    const int vertex_dofs, const int edge_dofs, const int face_dofs)
{
  const int dof_count = 4*vertex_dofs + 4*edge_dofs + face_dofs;
  DofMapPermuter output(dof_count);
  return output;
  // TODO: make DofPermuter store all possible permutations
  //       make permutations store indices of permutation on each cell

/*
  const int num_vertices = num_cell_vertices(type);
  std::cout << num_vertices << std::endl;

  const mesh::Geometry& geometry = mesh.geometry();
  const mesh::Topology& topology = mesh.topology();
  assert(topology.connectivity(dim, 0));
  const mesh::Connectivity& connectivity = *topology.connectivity(dim, 0);

  Eigen::ArrayXd h_cells = Eigen::ArrayXd::Zero(entities.rows());
  assert(num_vertices <= 8);
  std::array<Eigen::Vector3d, 8> points;
  for (Eigen::Index e = 0; e < entities.rows(); ++e)
  {
    // Get the coordinates  of the vertices
    const std::int32_t* vertices = connectivity.connections(entities[e]);
    for (int i = 0; i < num_vertices; ++i)
      points[i] = geometry.x(vertices[i]);

    // Get maximum edge length
    for (int i = 0; i < num_vertices; ++i)
    {
      for (int j = i + 1; j < num_vertices; ++j)
        h_cells[e] = std::max(h_cells[e], (points[i] - points[j]).norm());
    }
  }
*/

}
} // namespace fem
} // namespace dolfin
