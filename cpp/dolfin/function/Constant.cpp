// Copyright (C) 2019 Chris Richardson abd Michal Habera
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Constant.h"
#include <Eigen/Dense>
#include <vector>

using namespace dolfin;
using namespace dolfin::function;

//-----------------------------------------------------------------------------
Constant::Constant(PetscScalar value) : value({value})
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Constant::Constant(std::vector<PetscScalar> value, std::vector<int> shape)
    : value(value), shape(shape)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Constant::Constant(
    const Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        c)
    : value(c.rows() * c.cols()), shape({(int)c.rows(), (int)c.cols()})
{
  // Copy data from Eigen::Array to flattened vector
  for (int i = 0; i < c.rows(); ++i)
    for (int j = 0; j < c.cols(); ++j)
      value[i * c.cols() + j] = c(i, j);
}
//-----------------------------------------------------------------------------
