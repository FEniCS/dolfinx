// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-09
// Last changed: 2007-04-04

#include <dolfin/dolfin_log.h>
#include <dolfin/ConstantFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ConstantFunction::ConstantFunction(real value)
  : GenericFunction(), ufc::function(), value(value), size(1)
{
  cout << "Creating ConstantFunction" << endl;

  // Do nothing
}
//-----------------------------------------------------------------------------
ConstantFunction::~ConstantFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void ConstantFunction::interpolate(real* coefficients,
                                   const ufc::cell& cell,
                                   const ufc::finite_element& finite_element)
{
  cout << "Interpolating ConstantFunction" << endl;

  // Evaluate each dof to get coefficients for nodal basis expansion
  for (uint i = 0; i < finite_element.space_dimension(); i++)
    coefficients[i] = finite_element.evaluate_dof(i, *this, cell);

  // Compute size of value (number of entries in tensor value)
  size = 1;
  for (uint i = 0; i < finite_element.value_rank(); i++)
    size *= finite_element.value_dimension(i);
}
//-----------------------------------------------------------------------------
void ConstantFunction::evaluate(real* values,
                                const real* coordinates,
                                const ufc::cell& cell) const
{
  cout << "Evaluating ConstantFunction" << endl;

  // Set all values to the constant value
  for (uint i = 0; i < size; i++)
    values[i] = value;
}
//-----------------------------------------------------------------------------
