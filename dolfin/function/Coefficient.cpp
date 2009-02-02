// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Martin Alnes, 2008.
//
// First added:  2008-10-28
// Last changed: 2008-12-12

#include <sstream>

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
  boost::shared_ptr<Function> _v(reference_to_no_delete_pointer(v));
  attach(_v);
}
//-----------------------------------------------------------------------------
void Coefficient::attach(boost::shared_ptr<Function> v)
{
  // FIXME: This logic doesn't scale to the case where a Function is
  // FIXME: used as a coefficient two or more places. Confusing!

  // Set function space if not set
  if (!v->_function_space)
  {
    boost::shared_ptr<const FunctionSpace> _V(create_function_space());
    v->_function_space = _V;
  }

  // Set variable name
  if (v->label() == "unnamed function")
  {
    std::stringstream label;
    label << "coefficient " << number();
    v->rename(name(), label.str());
  }

  // Set coefficient for form
  dolfin_assert(number() < form._coefficients.size());
  form._coefficients[number()] = v;
}
//-----------------------------------------------------------------------------
