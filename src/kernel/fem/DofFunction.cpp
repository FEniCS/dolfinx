// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Vector.h>
#include <dolfin/ElementFunction.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/Node.h>
#include <dolfin/Point.h>
#include <dolfin/DofFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DofFunction::DofFunction(Vector& dofs) : x(dofs)
{

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
  return x(n.id());
}
//-----------------------------------------------------------------------------
real DofFunction::operator() (const Point& p, real t) const
{
  // FIXME: Needs implementation
  std::cout << "Evaluation of function at given point not implemented." << std::endl;
  exit(1);

  return 0.0;
}
//-----------------------------------------------------------------------------
