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

//-----------------------------------------------------------------------------
CoefficientAssigner::CoefficientAssigner(Form& form, std::size_t number)
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
  dolfin_assert(coefficient);
  _form.set_coefficient(_number, coefficient);
}
//-----------------------------------------------------------------------------
