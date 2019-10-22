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
  /// @param[in] mesh The mesh
  /// @param[in] element_dof_layout The layout of dofs in each cell
  DofMapPermuter(const mesh::Mesh mesh,
                 const ElementDofLayout& element_dof_layout);

  /// Return the dof permutations for the given cell
  /// @param[in] cell The cell index
  /// @return The permutation for the given cell
  std::vector<int> cell_permutation(const int cell) const;

private:
  // Functions called by the constructor for specific mesh types
  void _generate_triangle(const mesh::Mesh mesh,
                          const ElementDofLayout& element_dof_layout);
  void _generate_tetrahedron(const mesh::Mesh mesh,
                             const ElementDofLayout& element_dof_layout);
  void _generate_empty(const mesh::Mesh mesh,
                       const ElementDofLayout& element_dof_layout);

  void _resize_data();

  /// The number of dofs and cells and permutations
  int _dof_count;
  int _cell_count;
  int _permutation_count;

  /// Sets the orders of a permutation for a cell
  /// @param[in] cell The cell index
  /// @param[in] permutation The permutation index
  /// @param[in] orders The permutation order
  void _set_order(const int cell, const int permutation, const int order);

  /// The orders of each cell
  Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic> _cell_orders;

  /// The permutations
  Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic> _permutations;

  /// The orders of each permutation
  std::vector<int> _permutation_orders;

  /// Sets a permutation to the DofMapPermuter
  /// @param[in] index The index of the permutation
  /// @param[in] dofs The local dofs on the mesh element to be permuted
  /// @param[in] base_permutation The reordering of the local dofs
  /// @param[in] order The order of the permutation
  void _set_permutation(const int index,
                        const Eigen::Array<PetscInt, Eigen::Dynamic, 1> dofs,
                        const std::vector<int> base_permutation, int order);
};

} // namespace fem
} // namespace dolfin
