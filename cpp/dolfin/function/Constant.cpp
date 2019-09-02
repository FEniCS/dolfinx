// Copyright (C) 2019 Chris Richardson, Michal Habera
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Constant.h"
#include <vector>


using namespace dolfin;
using namespace dolfin::function;


//-----------------------------------------------------------------------------
Constant::Constant(std::vector<PetscScalar> value, std::vector<int> shape)
    : value(value), shape(shape){}
//-----------------------------------------------------------------------------
/// Initialise with a scalar value
Constant::Constant(PetscScalar value) : value({value}), shape({1}){}
//-----------------------------------------------------------------------------
/// Initialise with a 1D or 2D Array
Constant::Constant(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
              array)
    : shape({(int)array.rows(), (int)array.cols()})
{
  value.resize(array.rows() * array.cols());

  // Remove trailing 1 in shape for Eigen::Vector (1D array)
  if (array.cols() == 1)
    shape.pop_back();

  // Copy data from Eigen::Array to flattened vector
  Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>>
      v(value.data(), array.rows(), array.cols());
  v = array;
}
