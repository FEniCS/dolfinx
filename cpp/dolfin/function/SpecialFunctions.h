// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Expression.h"
#include <Eigen/Dense>
#include <memory>
#include <petscsys.h>

namespace dolfin
{
namespace mesh
{
class Cell;
class Mesh;
} // namespace mesh

namespace function
{

/// This Function represents the mesh coordinates on a given mesh.
class MeshCoordinates : public Expression
{
public:
  /// Constructor
  explicit MeshCoordinates(std::shared_ptr<const mesh::Mesh> mesh);

  /// Evaluate function
  void eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
                values,
            Eigen::Ref<const EigenRowArrayXXd> x, const mesh::Cell& cell) const;

private:
  // The mesh
  std::shared_ptr<const mesh::Mesh> _mesh;
};

/// This function represents the area/length of a cell facet on a
/// given mesh.
class FacetArea : public Expression
{
public:
  /// Constructor
  explicit FacetArea(std::shared_ptr<const mesh::Mesh> mesh);

  /// Evaluate function
  void eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
                values,
            Eigen::Ref<const EigenRowArrayXXd> x, const mesh::Cell& cell) const;

private:
  // The mesh
  std::shared_ptr<const mesh::Mesh> _mesh;
};
} // namespace function
} // namespace dolfin
