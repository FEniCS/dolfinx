// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-10-28
// Last changed: 2008-12-03

#include <dolfin/log/log.h>
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
  // Create function space
  std::tr1::shared_ptr<const FunctionSpace> _V(create_function_space());

  // Set function space if not set or check that it matches
  if (!v._function_space)
  {
    v._function_space = _V;
  }
  else
  {
    if (!v.in(*_V))
      error("Nonmatching function space for form coefficient.");
  }
  
  // Set variable name
  v.rename(name(), "A coefficient function");

  // Set coefficient for form
  dolfin_assert(number() < form._coefficients.size());
  std::tr1::shared_ptr<const Function> _v(&v, NoDeleter<const Function>());
  form._coefficients[number()] = _v;
}
//-----------------------------------------------------------------------------
