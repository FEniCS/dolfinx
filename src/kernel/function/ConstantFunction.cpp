// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-09
// Last changed: 2007-04-21

#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>
#include <dolfin/ConstantFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ConstantFunction::ConstantFunction(Mesh& mesh, real value)
  : GenericFunction(mesh), ufc::function(), value(value), size(1)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ConstantFunction::~ConstantFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint ConstantFunction::rank() const
{
  // Just return 0 for now (might extend to vectors later)
  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint ConstantFunction::dim(uint i) const
{
  // Just return 1 for now (might extend to vectors later)
  return 1;
}
//-----------------------------------------------------------------------------
void ConstantFunction::interpolate(real* values)
{
  dolfin_assert(values);

  // Set all vertex values to the constant value
  const uint num_values = size*mesh.numVertices();
  for (uint i = 0; i < num_values; i++)
    values[i] = value;
}
//-----------------------------------------------------------------------------
void ConstantFunction::interpolate(real* coefficients,
                                   const ufc::cell& cell,
                                   const ufc::finite_element& finite_element)
{
  dolfin_assert(coefficients);

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
  dolfin_assert(values);
  dolfin_assert(coordinates);

  // Set all values to the constant value
  for (uint i = 0; i < size; i++)
    values[i] = value;
}
//-----------------------------------------------------------------------------
