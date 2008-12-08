// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Martin Alnes, 2008.
//
// First added:  2008-10-28
// Last changed: 2008-12-08

#include <dolfin/common/NoDeleter.h>
#include <dolfin/fem/Form.h>
#include "Function.h"
#include "FunctionSpace.h"
#include "Coefficient.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Coefficient::Coefficient(Form& form) : form(form)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Coefficient::~Coefficient()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Coefficient::attach(Function& v)
{
  std::tr1::shared_ptr<Function> _v(reference_to_no_delete_pointer(v));
  attach(_v);
}
//-----------------------------------------------------------------------------
void Coefficient::attach(std::tr1::shared_ptr<Function> v)
{
  // Set function space if not set // TODO: This logic doesn't scale to the case where a Function is used as a coefficient two or more places. Confusing!
  if (!v->_function_space)
  {
    std::tr1::shared_ptr<const FunctionSpace> _V(create_function_space());
    v->_function_space = _V;
  }

  // Set variable name
  v->rename(name(), "A coefficient function"); // TODO: This logic doesn't scale to the case where a Function is used as a coefficient two or more places. Confusing!

  // Set coefficient for form
  dolfin_assert(number() < form._coefficients.size());
  form._coefficients[number()] = v;
}
//-----------------------------------------------------------------------------
