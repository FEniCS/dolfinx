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
  // Initialise as a vector
  explicit Constant(std::vector<PetscScalar> value) : value(value){};

  /// Value
  std::vector<PetscScalar> value;
};
} // namespace function
} // namespace dolfin
