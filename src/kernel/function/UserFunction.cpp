// Copyright (C) 2005-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-26
// Last changed: 2007-04-12
//
// Note: this breaks the standard envelope-letter idiom slightly,
// since we call the envelope class from one of the letter classes.

#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>
#include <dolfin/Vertex.h>
#include <dolfin/Function.h>
#include <dolfin/UserFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
UserFunction::UserFunction(Function* f)
  : GenericFunction(), ufc::function(), f(f), size(1)
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
void UserFunction::interpolate(real* values, Mesh& mesh)
{
  dolfin_assert(values);
  dolfin_assert(f);

  // Call overloaded eval function at each vertex
  real* local_values = new real[size];
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {
    // Evaluate at function at vertex
    f->eval(local_values, vertex->x());

    // Copy values to array of vertex values
    for (uint i = 0; i < size; i++)
      values[i*mesh.numVertices() + vertex->index()] = local_values[i];
  }
  delete [] local_values;
}
//-----------------------------------------------------------------------------
void UserFunction::interpolate(real* coefficients,
                               const ufc::cell& cell,
                               const ufc::finite_element& finite_element)
{
  dolfin_assert(coefficients);

  // Evaluate each dof to get coefficients for nodal basis expansion
  for (uint i = 0; i < finite_element.space_dimension(); i++)
    coefficients[i] = finite_element.evaluate_dof(i, *this, cell);
}
//-----------------------------------------------------------------------------
void UserFunction::evaluate(real* values,
                            const real* coordinates,
                            const ufc::cell& cell) const
{
  dolfin_assert(values);
  dolfin_assert(coordinates);
  dolfin_assert(f);

  // Call overloaded eval function
  f->eval(values, coordinates);
}
//-----------------------------------------------------------------------------
