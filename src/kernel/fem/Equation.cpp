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
  k = 0.0;
  
  this->dim = dim;
  
  noeq = 1; // Will be set otherwise by EquationSystem
}
//-----------------------------------------------------------------------------
Equation::~Equation()
{

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
void Equation::setTimeStep(real k)
{
  this->k = k;
}
//-----------------------------------------------------------------------------
real Equation::dx(real a) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real Equation::dy(real a) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real Equation::dz(real a) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real Equation::dt(real a) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
const ElementFunction& Equation::dx(const ShapeFunction &v) const
{
  return v.dx();
}
//-----------------------------------------------------------------------------
const ElementFunction& Equation::dy(const ShapeFunction &v) const
{
  return v.dy();
}
//-----------------------------------------------------------------------------
const ElementFunction& Equation::dz(const ShapeFunction &v) const
{
  return v.dz();
}
//-----------------------------------------------------------------------------
const ElementFunction& Equation::dt(const ShapeFunction &v) const
{
  return v.dt();
}
//-----------------------------------------------------------------------------
const ElementFunction Equation::dx(const Product &v) const
{
  return mapping->dx(v);
}
//-----------------------------------------------------------------------------
const ElementFunction Equation::dy(const Product &v) const
{
  return mapping->dy(v);
}
//-----------------------------------------------------------------------------
const ElementFunction Equation::dz(const Product &v) const
{
  return mapping->dz(v);
}
//-----------------------------------------------------------------------------
const ElementFunction Equation::dt(const Product &v) const
{
  return mapping->dt(v);
}
//-----------------------------------------------------------------------------
const ElementFunction Equation::dx(const ElementFunction &v) const
{
  return mapping->dx(v);
}
//-----------------------------------------------------------------------------
const ElementFunction Equation::dy(const ElementFunction &v) const
{
  return mapping->dy(v);
}
//-----------------------------------------------------------------------------
const ElementFunction Equation::dz(const ElementFunction &v) const
{
  return mapping->dz(v);
}
//-----------------------------------------------------------------------------
const ElementFunction Equation::dt(const ElementFunction &v) const
{
  return mapping->dt(v);
}
//-----------------------------------------------------------------------------
const FunctionSpace::Vector<ElementFunction>
Equation::grad(const ShapeFunction &v)
{
  FunctionSpace::Vector<ElementFunction> w(v.dx(), v.dy(), v.dz());

  return w;
}
//-----------------------------------------------------------------------------
void Equation::add(ElementFunction &v, Function &f)
{
  FunctionPair p(v, f);
  if ( functions.add(p) == -1 ) {
	 functions.resize(functions.size() + 1);
	 functions.add(p);
  }
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

  // Save mapping (to compute derivatives)
  this->mapping = mapping;
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
	 std::cout << "Assignment to int must be zero." << std::endl;
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
