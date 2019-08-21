// Copyright (C) 2019 Chris Richardson, Michal Habera
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <vector>

namespace dolfin
{

namespace function
{

class Constant
{
public:
  Constant(
      Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          value)
      : value(value){};

  /// Value
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      value;
};
} // namespace fem
} // namespace dolfin