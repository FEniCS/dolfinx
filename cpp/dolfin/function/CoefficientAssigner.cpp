// Copyright (C) 2008-2009 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CoefficientAssigner.h"
#include <dolfin/fem/Form.h>
#include <dolfin/log/log.h>
#include <memory>

using namespace dolfin;
using namespace dolfin::function;

//-----------------------------------------------------------------------------
CoefficientAssigner::CoefficientAssigner(fem::Form& form, std::size_t number)
    : _form(form), _number(number)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
CoefficientAssigner::~CoefficientAssigner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void CoefficientAssigner::
operator=(std::shared_ptr<const GenericFunction> coefficient)
{
  assert(coefficient);
  _form.coeffs().set(_number, coefficient);
}
//-----------------------------------------------------------------------------
