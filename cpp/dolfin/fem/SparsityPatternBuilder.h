// Copyright (C) 2007-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfin/la/SparsityPattern.h>

namespace dolfin
{
namespace la
{
class SparsityPattern;
}

namespace mesh
{
class Mesh;
}

namespace fem
{

/// This class provides functions to compute the sparsity pattern
/// based on DOF map data

class SparsityPatternBuilder
{
public:
  /// Iterate over cells and insert entries into sparsity pattern
  static void cells(
      la::SparsityPattern& pattern, const mesh::Mesh& mesh,
      const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& dofs0,
      int dim0,
      const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& dofs1,
      int dim1);

  /// Iterate over interior facets and insert entries into sparsity pattern
  static void interior_facets(
      la::SparsityPattern& pattern, const mesh::Mesh& mesh,
      const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& dofs0,
      int dim0,
      const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& dofs1,
      int dim1);

  /// Iterate over exterior facets and insert entries into sparsity pattern
  static void exterior_facets(
      la::SparsityPattern& pattern, const mesh::Mesh& mesh,
      const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& dofs0,
      int dim0,
      const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& dofs1,
      int dim1);
};
} // namespace fem
} // namespace dolfin
