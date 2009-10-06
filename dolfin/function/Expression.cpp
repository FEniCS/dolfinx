// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-28
// Last changed: 2009-10-06

// Modified by Johan Hake, 2009

#include <dolfin/log/log.h>
#include "Data.h"
#include "Expression.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Expression::Expression(uint geometric_dimension)
  : _geometric_dimension(geometric_dimension)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Expression::Expression(uint geometric_dimension, uint dim)
  : _geometric_dimension(geometric_dimension)
{
  value_shape.resize(1);
  value_shape[0] = dim;
}
//-----------------------------------------------------------------------------
Expression::Expression(uint geometric_dimension,
                       std::vector<uint> value_shape)
  : value_shape(value_shape), _geometric_dimension(geometric_dimension)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Expression::Expression(const Expression& expression)
  : value_shape(expression.value_shape),
    _geometric_dimension(expression._geometric_dimension)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Expression::~Expression()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint Expression::geometric_dimension() const
{
  return _geometric_dimension;
}
//-----------------------------------------------------------------------------
dolfin::uint Expression::value_rank() const
{
  return value_shape.size();
}
//-----------------------------------------------------------------------------
dolfin::uint Expression::value_dimension(uint i) const
{
  if (i >= value_shape.size())
    error("Illegal axis %d for value dimension for value of rank %d.",
          i, value_shape.size());
  return value_shape[i];
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
