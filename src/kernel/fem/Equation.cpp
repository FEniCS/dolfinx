// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Equation.h>
#include <dolfin/FiniteElement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Equation::Equation(int dim)
{
  h  = 0.0;
  t  = 0.0;
  dt = 0.0;
  
  this->dim = dim;
  noeq = 1; // Will be set otherwise by EquationSystem
}
//-----------------------------------------------------------------------------
void Equation::updateLHS(const FiniteElement* element,
								 const Cell*          cell,
								 const Mapping*       mapping,
								 const Quadrature*    quadrature)
{
  // Common update for LHS and RHS
  update(element, cell, mapping, quadrature);

  // Local update of LHS
  updateLHS();
}
//-----------------------------------------------------------------------------
void Equation::updateRHS(const FiniteElement* element,
								 const Cell*          cell,
								 const Mapping*       mapping,
								 const Quadrature*    quadrature)
{
  // Common update for LHS and RHS
  update(element, cell, mapping, quadrature);

  // Local update of RHS
  updateRHS();
}
//-----------------------------------------------------------------------------
void Equation::setTime(real t)
{
  this->t = t;
}
//-----------------------------------------------------------------------------
void Equation::setTimeStep(real dt)
{
  this->dt = dt;
}
//-----------------------------------------------------------------------------
void Equation::add(ElementFunction &v, Function &f)
{
  FunctionPair p(v, f);
  functions.add(p);
}
//-----------------------------------------------------------------------------
void Equation::update(const FiniteElement* element,
							 const Cell*          cell,
							 const Mapping*       mapping,
							 const Quadrature*    quadrature)
{
  // Update element functions
  for (ShortList<FunctionPair>::Iterator p(functions); !p.end(); ++p)
	 p->update(*element, *cell, t);

  // Update integral measures
  dK.update(*mapping, *quadrature);
  dS.update(*mapping, *quadrature);
}
//-----------------------------------------------------------------------------
// Equation::FunctionPair
//-----------------------------------------------------------------------------
Equation::FunctionPair::FunctionPair()
{
  v = 0;
  f = 0;
}
//-----------------------------------------------------------------------------
Equation::FunctionPair::FunctionPair(ElementFunction &v, Function &f)
{
  this->v = &v;
  this->f = &f;
}
//-----------------------------------------------------------------------------
void Equation::FunctionPair::update
(const FiniteElement &element, const Cell &cell, real t)
{
  f->update(*v, element, cell, t);
}
//-----------------------------------------------------------------------------
void Equation::FunctionPair::operator= (int zero)
{
  // FIXME: Use logging system
  if ( zero != 0 ) {
	 cout << "Assignment to int must be zero." << endl;
	 exit(1);
  }
  
  v = 0;
  f = 0;
}
//-----------------------------------------------------------------------------
bool Equation::FunctionPair::operator! () const
{
  return v == 0;
}
//-----------------------------------------------------------------------------
