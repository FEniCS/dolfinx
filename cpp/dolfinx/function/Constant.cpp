// Copyright (C) 2019 Chris Richardson and Michal Habera
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Constant.h"
#include <Eigen/Dense>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::function;

//-----------------------------------------------------------------------------
Constant::Constant(PetscScalar c) : value({c})
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Constant::Constant(const std::vector<PetscScalar>& c)
    : shape(1, c.size()), value({c})
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Constant::Constant(
    const Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>& c)
    : shape({(int)c.rows(), (int)c.cols()}), value(c.rows() * c.cols())
{
  // Copy data from Eigen::Array to flattened vector
  for (int i = 0; i < c.rows(); ++i)
    for (int j = 0; j < c.cols(); ++j)
      value[i * c.cols() + j] = c(i, j);
}
//-----------------------------------------------------------------------------
Constant::Constant(std::vector<int> shape, std::vector<PetscScalar> c)
    : shape(shape), value(c)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
