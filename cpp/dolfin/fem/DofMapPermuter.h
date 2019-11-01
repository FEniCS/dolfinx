// Copyright (C) 2019 Matthew Scroggs
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <vector>

namespace dolfin
{

namespace mesh
{
class Mesh;
} // namespace mesh

namespace fem
{
class ElementDofLayout;

/// A class to permute dofs on each cell
class DofMapPermuter
{
public:
  /// Constructor
  /// @param[in] mesh The mesh
  /// @param[in] dof_layout The layout of dofs on a each cell
  DofMapPermuter(const mesh::Mesh& mesh, const ElementDofLayout& dof_layout);

  /// Return the dof permutation for the given cell
  /// @param[in] cell The cell index
  /// @return The permutation for the given cell
  std::vector<int> get_cell_permutation(const int cell) const;

private:
  // Stores the number of times each row of _permutations should be applied on
  // each cell Will have shape (number of cells) × (number of permutations)
  Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic> _cell_ordering;

  // Each row of this represent the rotation or reflection of a mesh entity
  // Will have shape (number of permutations) × (number of dofs on reference)
  //   where (number of permutations) = (num_edges + 2*num_faces +
  //   4*num_volumes)
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic> _permutations;
};

} // namespace fem
} // namespace dolfin
