// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2006-02-09
// Last changed: 2008-07-08

#include <dolfin/log/dolfin_log.h>
#include <dolfin/mesh/Mesh.h>
#include "ConstantFunction.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ConstantFunction::ConstantFunction(const ConstantFunction& f)
  : GenericFunction(f.mesh), values(0), value_rank(f.value_rank), shape(0), size(f.size)
{
  values = new real[size];
  shape = new uint[value_rank];
  for(uint i=0; i<value_rank; i++)
  {
    shape[i] = f.shape[i];
  }
  for(uint i=0; i<size; i++)
  {
    values[i] = f.values[i];
  }
}
//-----------------------------------------------------------------------------
ConstantFunction::ConstantFunction(Mesh& mesh, real value)
  : GenericFunction(mesh), values(0), value_rank(0), shape(0), size(1)
{
  values = new real[1];
  shape = new uint[1];
  values[0] = value;
  shape[0] = 1;
}
//-----------------------------------------------------------------------------
ConstantFunction::ConstantFunction(Mesh& mesh, uint size, real value)
  : GenericFunction(mesh), values(0), value_rank(1), shape(0), size(size)
{
  shape = new uint[1];
  shape[0] = size;
  values = new real[size];
  for(uint i=0; i<size; i++)
  {
    values[i] = value;
  }
}
//-----------------------------------------------------------------------------
ConstantFunction::ConstantFunction(Mesh& mesh, const Array<real>& _values)
  : GenericFunction(mesh), values(0), value_rank(1), shape(0), size(0)
{
  size = _values.size();
  shape = new uint[1];
  shape[0] = size;
  values = new real[size];
  for(uint i=0; i<size; i++)
  {
    values[i] = _values[i];
  }
}
//-----------------------------------------------------------------------------
ConstantFunction::ConstantFunction(Mesh& mesh, const Array<uint>& _shape, const Array<real>& _values)
  : GenericFunction(mesh), values(0), value_rank(0), shape(0), size(0)
{
  value_rank = _shape.size();
  shape = new uint[value_rank];
  size = 1;
  for(uint i=0; i<value_rank; i++)
  {
    shape[i] = _shape[i];
    size *= shape[i];
  }
  if(size != _values.size())
    error("Size of given values does not match shape.");
  values = new real[size];
  for(uint i=0; i<size; i++)
  {
    values[i] = _values[i];
  }
}
//-----------------------------------------------------------------------------
ConstantFunction::~ConstantFunction()
{
  delete [] shape;
  delete [] values;
}
//-----------------------------------------------------------------------------
dolfin::uint ConstantFunction::rank() const
{
  return value_rank;
}
//-----------------------------------------------------------------------------
dolfin::uint ConstantFunction::dim(uint i) const
{
  if(i >= value_rank)
    error("Too large dimension in dim.");
  return shape[i];
}
//-----------------------------------------------------------------------------
void ConstantFunction::interpolate(real* _values) const
{
  dolfin_assert(_values);

  // Set all vertex values to the constant tensor value
  for (uint i = 0; i < mesh.numVertices(); i++)
  {
    for (uint j = 0; j < size; j++)
    {
      uint k = i*size + j;
      _values[k] = values[j];
    }
  }
}
//-----------------------------------------------------------------------------
void ConstantFunction::interpolate(real* coefficients,
                                   const ufc::cell& cell,
                                   const ufc::finite_element& finite_element) const
{
  dolfin_assert(coefficients);
  
  // Assert same value shape (TODO: Slow to do this for every element, should probably remove later)
  dolfin_assert(value_rank == finite_element.value_rank());
  for (uint i = 0; i < value_rank; i++)
    dolfin_assert(shape[i] == finite_element.value_dimension(i));
  
  // UFC 1.0 version:
  // Evaluate each dof to get coefficients for nodal basis expansion
  for (uint i = 0; i < finite_element.space_dimension(); i++)
    coefficients[i] = finite_element.evaluate_dof(i, *this, cell);
  
  // UFC 1.1 version:
  /// Evaluate linear functionals for all dofs on the function f
  //finite_element.evaluate_dofs(coefficients, *this, cell);
}
//-----------------------------------------------------------------------------
void ConstantFunction::eval(real* _values, const real* x) const
{
  dolfin_assert(_values);

  // Set all values to the constant tensor value
  for (uint i = 0; i < size; i++)
    _values[i] = values[i];
}
//-----------------------------------------------------------------------------
void ConstantFunction::evaluate(real* _values,
                                const real* coordinates,
                                const ufc::cell& cell) const
{
  // Call eval(), cell ignored
  eval(_values, coordinates);
}
//-----------------------------------------------------------------------------
