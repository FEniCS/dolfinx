// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Assembler.h"
#include "DirichletBC.h"
#include "Form.h"
#include <string>

using namespace dolfin;

//-----------------------------------------------------------------------------
Assembler::Assembler(std::shared_ptr<const Form> a,
                     std::shared_ptr<const Form> L,
                     std::vector<std::shared_ptr<const DirichletBC>> bcs)
    : _a(a), _l(L), _bcs(bcs)
{
  // Check rank of forms
  if (a and a->rank() != 2)
  {
    throw std::runtime_error(
        "Expecting bilinear form (rank 2), but form has rank \'"
        + std::to_string(a->rank()) + "\'");
  }
  if (L and L->rank() != 1)
  {
    throw std::runtime_error(
        "Expecting linear form (rank 1), but form has rank \'"
        + std::to_string(L->rank()) + "\'");
  }
}
//-----------------------------------------------------------------------------