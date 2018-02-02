// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <memory>
#include <utility>
#include <vector>

namespace dolfin
{

// Forward declarations
class DirichletBC;
class Form;

namespace fem
{

class Assembler
{
public:
  /// Constructor
  Assembler(std::shared_ptr<const Form> a, std::shared_ptr<const Form> L,
            std::vector<std::shared_ptr<const DirichletBC>> bcs);

private:
  // Bilinear and linear forms
  std::shared_ptr<const Form> _a, _l;

  // Boundary conditions
  std::vector<std::shared_ptr<const DirichletBC>> _bcs;
};
}
}
