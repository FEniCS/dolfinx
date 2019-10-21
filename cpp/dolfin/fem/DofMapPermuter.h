// Copyright (C) 2019 Matthew Scroggs
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ElementDofLayout.h"
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
/// A class to permute dofs on each cell
class DofMapPermuter
{
public:
  /// Constructor
  DofMapPermuter();

  /// Return the dof permutations for the given cell
  /// @param[in] cell The cell index
  /// @return The permutation for the given cell
  std::vector<int> cell_permutation(const int cell) const;

  /// Permute a vector
  /// @param[in] vec A vector to permute
  /// @param[in] perm A permutation
  /// @return The permuted vector
  std::vector<int> permute(std::vector<int> vec, std::vector<int> perm) const;

  /// Adds a permutation to the DofMapPermuter
  /// @param[in] permutation The permutation
  /// @param[in] order The order of the permutation
  void add_permutation(const std::vector<int> permutation, int order);

  /// Sets the orders of each permutation for a cell
  /// @param[in] cell The cell index
  /// @param[in] orders The permutation orders
  void set_cell(const int cell, const std::vector<int> orders);

  /// Sets the number of cells
  /// @param[in] cells Number of cells
  void set_cell_count(const int cells);

  /// Sets the number of dofs per cell
  /// @param[in] cells Number of dofs per cell
  void set_dof_count(const int dofs);

  /// Calculates the permutation orders for a triangle
  /// @param[in] v1 The global vertex number of the triangle's first vertex
  /// @param[in] v2 The global vertex number of the triangle's second vertex
  /// @param[in] v3 The global vertex number of the triangle's third vertex
  /// @return The rotation and reflection orders for the triangle
  std::pair<int, int> calculate_triangle_orders(int v1, int v2, int v3);

  /// Calculates the permutation orders for a tetrahedron
  /// @param[in] v1 The global vertex number of the tetrahedron's first vertex
  /// @param[in] v2 The global vertex number of the tetrahedron's second vertex
  /// @param[in] v3 The global vertex number of the tetrahedron's third vertex
  /// @param[in] v4 The global vertex number of the tetrahedron's fourth vertex
  /// @return The rotation and reflection orders for the tetrahedron
  std::tuple<int, int, int, int> calculate_tetrahedron_orders(int v1, int v2,
                                                              int v3, int v4);

private:
  /// The number of dofs
  int _dof_count;
  int _cell_count;

  /// The orders of each cell
  Eigen::Array<PetscInt, Eigen::Dynamic, 2> _cell_orders;

  /// The permutations
  std::vector<std::vector<int>> _permutations;

  /// The orders of each permutation
  std::vector<int> _permutation_orders;
};

/// Make the DofMapPermuter for a given mesh and dof layout
/// @param[in] mesh The mesh
/// @param[in] element_dof_layout The layout of dofs in each cell
/// @return A DofMapPermuter for the mesh and dof layout
DofMapPermuter
generate_cell_permutations(const mesh::Mesh mesh,
                           const ElementDofLayout& element_dof_layout);

} // namespace fem
} // namespace dolfin
