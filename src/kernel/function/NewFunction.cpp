// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Mesh.h>
#include <dolfin/NewFiniteElement.h>
#include <dolfin/NewVector.h>
#include <dolfin/NewFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewFunction::NewFunction(const Mesh& mesh, const NewFiniteElement& element,
			 NewVector& x) : mesh(mesh), element(element), x(x)
{
  rename("u", "An unspecified function");
}
//-----------------------------------------------------------------------------
NewFunction::~NewFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewFunction::project(const Cell& cell, const NewFiniteElement& element,
			  real c[]) const
{
  // Check if we're computing the projection onto a cell of the same
  // mesh for the same element (the easy case...)
  if ( &(cell.mesh()) == &mesh && &element == &(this->element) )
  {
    // FIXME: Assumes uniprocessor case. Why isn't there a function
    // FIXME: VecGetValues() in PETSc? Possible answer: since if we're
    // FIXME: doing this in parallel we only want to access this
    // FIXME: processor's data anyway.

    // FIXME: If we know that the values are stored element-by-element
    // FIXME: in x, then we can optimize by just calling
    // FIXME: element::dof() one time with i = 0.

    real *values = x.array();
    for (uint i = 0; i < element.spacedim(); i++)
      c[i] = values[element.dof(i, cell, mesh)];
    x.restore(values);
  }
  else
  {
    dolfin_error("Projection between different finite element spaces not implemented.");
  }
}
//-----------------------------------------------------------------------------
real NewFunction::operator()(const Point& p)
{
  // FIXME: Empty implementation. We need to decide what the default
  // behavior should be.

  return 0.0;
}
