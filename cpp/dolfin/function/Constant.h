// Copyright (C) 2019 Chris Richardson, Michal Habera
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <vector>
#include <dolfin/common/types.h>


namespace dolfin
{

namespace function
{

class Constant
{
public:
  /// Initialise as a vector
  ///
  /// The vector is a row-major (C style) flattened value of the constant
  Constant(std::vector<PetscScalar> value, std::vector<int> shape);

  /// Initialise with a scalar value
  Constant(PetscScalar value);

  /// Initialise with a 1D or 2D Array
  Constant(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
               array);
  /// Value
  std::vector<PetscScalar> value;

  /// Shape
  std::vector<int> shape;
};
} // namespace function
} // namespace dolfin
