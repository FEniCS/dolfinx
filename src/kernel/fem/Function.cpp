// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Settings.h>
#include <dolfin/Vector.h>
#include <dolfin/Point.h>
#include <dolfin/Cell.h>
#include <dolfin/Grid.h>
#include <dolfin/ElementFunction.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/Function.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Function::Function(Grid &grid, Vector &x)
{
  representation = DOF;
  
  this->x = &x;
  f = 0;
}
//-----------------------------------------------------------------------------
Function::Function(Grid &grid, const char *function)
{
  representation = FUNCTION;
  
  x = 0;
  Settings::get(function, &f);
}
//-----------------------------------------------------------------------------
void Function::update(FunctionSpace::ElementFunction& v,
							 const FiniteElement& element,
							 const Cell& cell, real t) const
{
  // Update degrees of freedom for element function, assuming it belongs to
  // the local trial space of the finite element.

  // Set dimension of function space for element function
  v.init(element.dim());

  // Update coefficients
  switch ( representation ) {
  case DOF:

	 // Set coefficients of element function from global degrees of freedom
	 for (FiniteElement::TrialFunctionIterator phi(element); !phi.end(); ++phi)
		v.set(phi.index(), phi, (*x)(phi.dof(cell)));
	 
	 break;
  case FUNCTION:

	 // Set coefficients of element function from function values
	 for (FiniteElement::TrialFunctionIterator phi(element); !phi.end(); ++phi)
		v.set(phi.index(), phi, phi.dof(cell, f, t));
	 
	 break;
  default:
	 // FIXME: Use logging system
	 cout << "Error: Unknown representation of function." << endl;
	 exit(1);
  }
}
//-----------------------------------------------------------------------------
