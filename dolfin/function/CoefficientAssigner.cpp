// Copyright (C) 2008-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Martin Alnes, 2008.
//
// First added:  2008-10-28
// Last changed: 2009-10-04

#include <boost/shared_ptr.hpp>
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
void CoefficientAssigner::operator= (const GenericFunction& coefficient)
{
  boost::shared_ptr<const GenericFunction> c(&coefficient, NoDeleter());
  form.set_coefficient(number, c);
}
//-----------------------------------------------------------------------------
