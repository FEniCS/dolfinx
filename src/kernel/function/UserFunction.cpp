// Copyright (C) 2005-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-26
// Last changed: 2007-04-12
//
// Note: this breaks the standard envelope-letter idiom slightly,
// since we call the envelope class from one of the letter classes.

#include <dolfin/Function.h>
#include <dolfin/UserFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
UserFunction::UserFunction(Function* f)
  : GenericFunction(), ufc::function(), f(f)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
UserFunction::~UserFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint UserFunction::rank() const
{
  // Just return 0 for now (might extend to vectors later)
  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint UserFunction::dim(uint i) const
{
  // Just return 1 for now (might extend to vectors later)
  return 1;
}
//-----------------------------------------------------------------------------
void UserFunction::interpolate(real* coefficients,
                               const ufc::cell& cell,
                               const ufc::finite_element& finite_element)
{
  // Evaluate each dof to get coefficients for nodal basis expansion
  for (uint i = 0; i < finite_element.space_dimension(); i++)
    coefficients[i] = finite_element.evaluate_dof(i, *this, cell);
}
//-----------------------------------------------------------------------------
void UserFunction::evaluate(real* values,
                            const real* coordinates,
                            const ufc::cell& cell) const
{
  // Call overloaded eval function
  f->eval(values, coordinates);
}
//-----------------------------------------------------------------------------
