// Copyright (C) 2019 Matthew Scroggs
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <tuple>
#include <vector>
#include <iostream>  // REMOVE: for testing only

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
  std::vector<int> cell_permutation(const int cell) const;
  std::vector<int> permute(std::vector<int> vec, std::vector<int> perm) const;
  void add_permutation(const std::vector<int> permutation, int order);

  void set_cell(const int cell,const int permutation);
  void set_cell(const int cell, const std::vector<int> orders);
  void prepare(const int cells);
  int get_permutation_number(const std::vector<int> orders) const;
  std::vector<int> get_orders(const int number) const;

  // REMOVE: for testing only
  void print_p(const std::vector<int> p, const int start=0) const {
    std::cout << "["; for(int i=start;i<p.size();++i) {std::cout << p[i];if(i+1<p.size()) std::cout << " ";}
    std::cout << "]" << std::endl;
  }

  int dof_count;
private:
  std::vector<std::vector<int>> _cell_orders;

  std::vector<std::vector<int>> _permutations;
  std::vector<int> _permutation_orders;
  int _total_options;
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
