// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-28
// Last changed: 2009-10-03

#include <dolfin/log/log.h>
#include "Data.h"
#include "Expression.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Expression::Expression()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Expression::~Expression()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Expression::eval(double* values, const double* x) const
{
  assert(values);
  assert(x);

  // Missing eval method if we reach this point
  error("Missing eval() for expression (must be overloaded).");
}
//-----------------------------------------------------------------------------
void Expression::eval(double* values, const Data& data) const
{
  assert(values);
  assert(data.is_valid());

  // Redirect to simple eval
  eval(values, data.x);
}
//-----------------------------------------------------------------------------
void Expression::restrict(double* w,
                          const FiniteElement& element,
                          const Cell& dolfin_cell,
                          const ufc::cell& ufc_cell,
                          int local_facet) const
{
  // Restrict as UFC function (by calling eval)
  restrict_as_ufc_function(w, element, dolfin_cell, ufc_cell, local_facet);
}
//-----------------------------------------------------------------------------
