// Copyright (C) 2019 Chris Richardson and Michal Habera
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfin/common/types.h>
#include <vector>

namespace dolfin
{

namespace function
{

class Constant
{
public:
  /// Create a rank-0 (scalar-valued) constant
  Constant(PetscScalar c);

  /// Create a rank-1 (vector-valued) constant
  Constant(std::vector<PetscScalar> c);

  /// Create a rank-2 constant
  Constant(const Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                         Eigen::Dynamic, Eigen::RowMajor>>
               c);

  /// The arbitrary rank constant. Data layout is row-major (C style).
  Constant(std::vector<int> shape, std::vector<PetscScalar> value);

  /// Shape
  std::vector<int> shape;

  /// Values
  std::vector<PetscScalar> value;
};
} // namespace function
} // namespace dolfin
