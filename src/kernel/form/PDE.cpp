// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Fredrik Bengzon and Johan Jansson, 2004.

#include <dolfin/dolfin_log.h>
#include <dolfin/PDE.h>
#include <dolfin/FiniteElement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PDE::PDE(int dim, int noeq) : dim(dim), noeq(noeq)
{
  h  = 0.0;
  t  = 0.0;
  k = 0.0;
}
//-----------------------------------------------------------------------------
PDE::~PDE()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const void PDE::updateLHS(FiniteElement::Vector& element,
			  //		    const Cell& cell,
			  const Map& map,
			  const Quadrature& interior_quadrature,
			  const Quadrature& boundary_quadrature)
{
  // Common update for LHS and RHS
  update(element, map, interior_quadrature, boundary_quadrature);
  
  // Local update of LHS
  updateLHS();
}
//-----------------------------------------------------------------------------
const void PDE::updateRHS(FiniteElement::Vector& element,
			  //		    const Cell& cell,
			  const Map& map,
			  const Quadrature& interior_quadrature,
			  const Quadrature& boundary_quadrature)
{
  // Common update for LHS and RHS
  update(element, map, interior_quadrature, boundary_quadrature);

  // Local update of RHS
  updateRHS();
}
//-----------------------------------------------------------------------------
real PDE::ddx(real a) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real PDE::ddy(real a) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real PDE::ddz(real a) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real PDE::ddt(real a) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
const ElementFunction& PDE::ddx(const ShapeFunction &v) const
{
  return v.ddx();
}
//-----------------------------------------------------------------------------
const ElementFunction& PDE::ddy(const ShapeFunction &v) const
{
  return v.ddy();
}
//-----------------------------------------------------------------------------
const ElementFunction& PDE::ddz(const ShapeFunction &v) const
{
  return v.ddz();
}
//-----------------------------------------------------------------------------
const ElementFunction& PDE::ddt(const ShapeFunction &v) const
{
  return v.ddt();
}
//-----------------------------------------------------------------------------
const ElementFunction PDE::ddx(const Product &v) const
{
  return map_->ddx(v);
}
//-----------------------------------------------------------------------------
const ElementFunction PDE::ddy(const Product &v) const
{
  return map_->ddy(v);
}
//-----------------------------------------------------------------------------
const ElementFunction PDE::ddz(const Product &v) const
{
  return map_->ddz(v);
}
//-----------------------------------------------------------------------------
const ElementFunction PDE::ddt(const Product &v) const
{
  return map_->ddt(v);
}
//-----------------------------------------------------------------------------
const ElementFunction PDE::ddx(const ElementFunction &v) const
{
  return map_->ddx(v);
}
//-----------------------------------------------------------------------------
const ElementFunction PDE::ddy(const ElementFunction &v) const
{
  return map_->ddy(v);
}
//-----------------------------------------------------------------------------
const ElementFunction PDE::ddz(const ElementFunction &v) const
{
  return map_->ddz(v);
}
//-----------------------------------------------------------------------------
const ElementFunction PDE::ddt(const ElementFunction &v) const
{
  return map_->ddt(v);
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction::Vector
PDE::grad(const ShapeFunction &v)
{
  FunctionSpace::ElementFunction::Vector w(v.ddx(), v.ddy(), v.ddz());
  return w;
}
//-----------------------------------------------------------------------------
void PDE::add(ElementFunction& v, Function& f)
{
  FunctionPair p(v, f);
  functions.add(p);
}
//-----------------------------------------------------------------------------
void PDE::add(ElementFunction::Vector& v, Function::Vector& f)
{
  if (v.size() != f.size())
  {
    dolfin_error("Function dimensions don't match.");
    dolfin_error2("v.size() = %d, f.size() = %d", v.size(), f.size());
  }

  for(int i = 0; i < f.size(); i++)
    add(v(i), f(i));
}
//-----------------------------------------------------------------------------
const void PDE::update(FiniteElement::Vector& element,
		       //                 const Cell& cell,
		       const Map& map,
		       const Quadrature& interior_quadrature,
		       const Quadrature& boundary_quadrature)
{
  // Update element functions
  // We assume that the element dependency is only on the grid, therefore
  // any element, such as the 0th is sufficient
  
  for (List<FunctionPair>::Iterator p(functions); !p.end(); ++p)
    p->update(*(element(0)), *(map.cell()), t);
  
  // Update integral measures
  dx.update(map, interior_quadrature);
  ds.update(map, boundary_quadrature);
  h = map.cell()->diameter();
  
  // Save map (to compute derivatives)
  this->map_ = &map;

  // Save cell 
  this->cell_ = map.cell();
}
//-----------------------------------------------------------------------------
int PDE::size()
{
  return noeq;
}
//-----------------------------------------------------------------------------
// PDE::FunctionPair
//-----------------------------------------------------------------------------
PDE::FunctionPair::FunctionPair()
{
  v = 0;
  f = 0;
}
//-----------------------------------------------------------------------------
PDE::FunctionPair::FunctionPair(ElementFunction &v, Function &f)
{
  this->v = &v;
  this->f = &f;
}
//-----------------------------------------------------------------------------
void PDE::FunctionPair::update
(const FiniteElement &element, const Cell &cell, real t)
{  
  // How do we do this for vector valued elements?

  f->update(*v, element, cell, t);
}
//-----------------------------------------------------------------------------
