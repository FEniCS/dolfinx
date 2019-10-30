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
  static Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic>
  generate_recursive(const mesh::Mesh& mesh,
                     const ElementDofLayout& element_dof_layout);

  // Functions called by the constructor for specific mesh types
  static Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic>
  generate_triangle(const mesh::Mesh& mesh,
                    const ElementDofLayout& element_dof_layout);

  static Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic>
  generate_tetrahedron(const mesh::Mesh& mesh,
                       const ElementDofLayout& element_dof_layout);

  // Functions called by the constructor for specific mesh types
  static Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic>
  set_orders_triangle(const mesh::Mesh& mesh,
                      const ElementDofLayout& element_dof_layout);
  static Eigen::Array<PetscInt, Eigen::Dynamic, Eigen::Dynamic>
  set_orders_tetrahedron(const mesh::Mesh& mesh,
                         const ElementDofLayout& element_dof_layout);

  // The orders of each cell
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic> _cell_orders;

  // The permutations
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic> _permutations;
};

} // namespace fem
} // namespace dolfin
