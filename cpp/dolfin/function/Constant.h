// Copyright (C) 2019 Chris Richardson, Michal Habera
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <vector>

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
  Constant(const std::vector<PetscScalar>& value, const std::vector<int>& shape)
      : value(value), shape(shape){};

  /// Initialise with a scalar
  Constant(PetscScalar value) : value({value}), shape({1}){};

  /// Initialise with an array
  Constant(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
               array)
      : shape({(int)array.rows(), (int)array.cols()})
  {
    value.resize(array.rows() * array.cols());
    Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>>
        v(value.data(), array.rows(), array.cols());
    v = array;
  }

  /// Value
  std::vector<PetscScalar> value;

  /// Shape
  std::vector<int> shape;
};
} // namespace function
} // namespace dolfin
