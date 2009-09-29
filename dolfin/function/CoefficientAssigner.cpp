// Copyright (C) 2008-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Martin Alnes, 2008.
//
// First added:  2008-10-28
// Last changed: 2009-09-29

#include <dolfin/common/NoDeleter.h>
#include <dolfin/fem/Form.h>
#include "CoefficientAssigner.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
CoefficientAssigner::CoefficientAssigner(Form& form, uint number)
  : form(form), number(number)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
CoefficientAssigner::~CoefficientAssigner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void CoefficientAssigner::operator= (const Coefficient& coefficient)
{
  assert(number < form._coefficients.size());
  form._coefficients[number] = reference_to_no_delete_pointer(coefficient);
}
//-----------------------------------------------------------------------------
