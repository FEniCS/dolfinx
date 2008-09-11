// Copyright (C) 2005-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-11-26
// Last changed: 2008-03-17
//
// Note: this breaks the standard envelope-letter idiom slightly,
// since we call the envelope class from one of the letter classes.

#include <dolfin/log/dolfin_log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include "Function.h"
#include "UserFunction.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UserFunction::UserFunction(Mesh& mesh, Function* f)
  : GenericFunction(mesh), ufc::function(), f(f)
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
  // Just return 0 (if not overloaded by user)
  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint UserFunction::dim(uint i) const
{
  // Just return 1 (if not overloaded by user)
  return 1;
}
//-----------------------------------------------------------------------------
void UserFunction::interpolate(real* values) const
{
  dolfin_assert(values);
  dolfin_assert(f);

  // Compute size of value (number of entries in tensor value)
  uint size = 1;
  for (uint i = 0; i < f->rank(); i++)
    size *= f->dim(i);

  // Call overloaded eval function at each vertex
  simple_array<real> local_values(size, new real[size]);
  
  for (VertexIterator vertex(*mesh); !vertex.end(); ++vertex)
  {
    // Evaluate at function at vertex
    simple_array<real> x(mesh->geometry().dim(), vertex->x());
    f->eval(local_values, x);

    // Copy values to array of vertex values
    for (uint i = 0; i < size; i++)
      values[i*mesh->numVertices() + vertex->index()] = local_values[i];
  }
  delete [] local_values.data;
}
//-----------------------------------------------------------------------------
void UserFunction::interpolate(real* coefficients,
                               const ufc::cell& cell,
                               const ufc::finite_element& finite_element) const
{
  dolfin_assert(coefficients);

  // Evaluate each dof to get coefficients for nodal basis expansion
  for (uint i = 0; i < finite_element.space_dimension(); i++)
    coefficients[i] = finite_element.evaluate_dof(i, *this, cell);
}
//-----------------------------------------------------------------------------
void UserFunction::eval(real* values, const real* x) const
{
  message("Calling user function");

  // Call user-overloaded eval function in Function
  f->eval(values, x);
}
//-----------------------------------------------------------------------------
void UserFunction::evaluate(real* values,
                            const real* coordinates,
                            const ufc::cell& cell) const
{
  dolfin_assert(values);
  dolfin_assert(coordinates);
  dolfin_assert(f);

  // Compute size of value (number of entries in tensor value)
  uint size = 1;
  for (uint i = 0; i < f->rank(); i++)
    size *= f->dim(i);

  // Call user-overloaded eval function in Function
  simple_array<real> v(size, values);
  simple_array<real> x(cell.geometric_dimension, const_cast<real*>(coordinates));
  f->eval(v, x);
}
//-----------------------------------------------------------------------------
