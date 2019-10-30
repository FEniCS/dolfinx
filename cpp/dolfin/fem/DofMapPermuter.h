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
  /// @param[in] dof_layout The layout of dofs on a each cell
  DofMapPermuter(const mesh::Mesh& mesh, const ElementDofLayout& dof_layout);

  /// Return the dof permutations for the given cell
  /// @param[in] cell The cell index
  /// @return The permutation for the given cell
  std::vector<int> cell_permutation(const int cell) const;

private:
  // Ordering on each cell
  Eigen::Array<std::int8_t, Eigen::Dynamic, Eigen::Dynamic> _cell_ordering;

  // The permutations
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic> _permutations;
};

} // namespace fem
} // namespace dolfin
