// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Grid.h>
#include <dolfin/Vector.h>
#include <dolfin/ElementFunction.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/Node.h>
#include <dolfin/Point.h>
#include <dolfin/DofFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DofFunction::DofFunction(Grid& grid, Vector& dofs, int dim, int size) :
  x(dofs), GenericFunction(dim, size)
{
  // FIXME: assumes nodal basis
  x.init(grid.noNodes());

  dolfin_debug1("adress = 0x%x", &x);
  x(0) = 0.0;
}
//-----------------------------------------------------------------------------
void DofFunction::update(FunctionSpace::ElementFunction &v,
			 const FiniteElement &element,
			 const Cell &cell,
			 real t) const
{
  for (FiniteElement::TrialFunctionIterator phi(element); !phi.end(); ++phi)
    v.set(phi.index(), phi, x(phi.dof(cell)));
}
//-----------------------------------------------------------------------------
real DofFunction::operator() (const Node& n, real t)  const
{
  // FIXME: assumes nodal basis
  dolfin_debug1("adress = 0x%x", &x);
  dolfin_debug1("size   = %d", x.size());
  dolfin_debug1("index  = %d", n.id()*size + dim);
  x(1) = 0.0;

  return x(n.id()*size + dim);
}
//-----------------------------------------------------------------------------
real DofFunction::operator() (const Point& p, real t) const
{
  dolfin_error("Evaluation of function at given point not implemented.");
  
  return 0.0;
}
//-----------------------------------------------------------------------------
