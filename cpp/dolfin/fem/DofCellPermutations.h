// Copyright (C) 2019 Matthew Scroggs
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <tuple>
#include <vector>

namespace dolfin
{

namespace mesh
{
class Mesh;
} // namespace mesh

namespace fem
{
/// Permutes dofs on each cell
class DofMapPermuter
{
public:
  /// Build permuter
  DofMapPermuter(const int dofs);
  const std::vector<int> permute(const std::vector<int> dofs, const int cell);
  int get_dof(const int cell, const int dof) const;
  const std::vector<int> permute(std::vector<int> vec, std::vector<int> perm);
  void add_edge_flip(const std::vector<int> flip);
  void set_reflection(const std::vector<int> reflection);
  void set_rotation(const std::vector<int> rotation, int order);
  void set_cell_permutation(const int cell,const int permutation);
  void set_cell_permutation(const int cell,const int rotations, const int reflections, const std::vector<int> edge_flips={});
  void prepare(const int cells);
  const int get_permutation_number(const int rotations, const int reflections, const std::vector<int> edge_flips={});
  void generate_necessary_permutations();

  // TODO: remove this; it's only for testing
  std::vector<int> p(){ return _permutations_of_cells;}
  int dof_count;
private:
  std::vector<int> _permutations_of_cells;
  std::vector<std::vector<int>> _permutations;
  std::vector<std::vector<int>> _edge_flips;
  std::vector<int> _reflection;
  std::vector<int> _rotation;
  int _rotation_order;
  int _total_options;
  std::vector<bool> _used;
};


DofMapPermuter generate_cell_permutations(const mesh::Mesh mesh,
                                          const int vertex_dofs,
                                          const int edge_dofs,
                                          const int face_dofs,
                                          const int volume_dofs);
DofMapPermuter generate_cell_permutations_triangle(
    const mesh::Mesh mesh, const int v, const int e, const int f);
DofMapPermuter generate_cell_permutations_quadrilateral(
    const mesh::Mesh mesh, const int v, const int e, const int f);
} // namespace fem
} // namespace dolfin
