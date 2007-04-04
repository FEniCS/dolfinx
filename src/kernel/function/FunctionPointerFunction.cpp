// Copyright (C) 2005-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-28
// Last changed: 2007-04-05

#include <dolfin/FunctionPointerFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionPointerFunction::FunctionPointerFunction(FunctionPointer f)
  : GenericFunction(), f(f)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionPointerFunction::~FunctionPointerFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void FunctionPointerFunction::interpolate(real* coefficients,
                                          const ufc::cell& cell,
                                          const ufc::finite_element& finite_element)
{
  // Evaluate each dof to get coefficients for nodal basis expansion
  for (uint i = 0; i < finite_element.space_dimension(); i++)
    coefficients[i] = finite_element.evaluate_dof(i, *this, cell);
}
//-----------------------------------------------------------------------------
void FunctionPointerFunction::evaluate(real* values,
                                       const real* coordinates,
                                       const ufc::cell& cell) const
{
  // Call function
  f(values, coordinates);
}
//-----------------------------------------------------------------------------
