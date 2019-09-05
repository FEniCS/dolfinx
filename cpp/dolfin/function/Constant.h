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
  /// Initialise with a scalar value
  Constant(PetscScalar value);

  /// Initialise with a 1D or 2D Array
  Constant(const Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                         Eigen::Dynamic, Eigen::RowMajor>>
               c);

  /// Initialise as a vector
  ///
  /// The vector is a row-major (C style) flattened value of the constant
  Constant(std::vector<int> shape, std::vector<PetscScalar> value);

  /// Shape
  std::vector<int> shape;

  /// Values
  std::vector<PetscScalar> value;
};
} // namespace function
} // namespace dolfin
