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
  DofMapPermuter(const mesh::Mesh& mesh,
                 const ElementDofLayout& element_dof_layout);

  /// Return the dof permutations for the given cell
  /// @param[in] cell The cell index
  /// @return The permutation for the given cell
  std::vector<int> cell_permutation(const int cell) const;

private:
  Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic>
  _generate_recursive(const mesh::Mesh& mesh,
                      const ElementDofLayout& element_dof_layout);

  // Functions called by the constructor for specific mesh types
  Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic>
  _generate_triangle(const mesh::Mesh& mesh,
                     const ElementDofLayout& element_dof_layout);

  Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic>
  _generate_tetrahedron(const mesh::Mesh& mesh,
                        const ElementDofLayout& element_dof_layout);

  void _set_orders(const mesh::Mesh& mesh,
                   const ElementDofLayout& element_dof_layout);

  // Functions called by the constructor for specific mesh types
  void _set_orders_triangle(const mesh::Mesh& mesh,
                            const ElementDofLayout& element_dof_layout);
  void _set_orders_tetrahedron(const mesh::Mesh& mesh,
                               const ElementDofLayout& element_dof_layout);

  // The number of dofs and cells and permutations
  int _dof_count;
  int _cell_count;
  int _permutation_count;

  // Sets the orders of a permutation for a cell
  // @param[in] cell The cell index
  /// @param[in] permutation The permutation index
  // @param[in] orders The permutation order
  void _set_order(const int cell, const int permutation, const int order);

  // The orders of each cell
  Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic> _cell_orders;

  // The permutations
  Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic> _permutations;
};

} // namespace fem
} // namespace dolfin
