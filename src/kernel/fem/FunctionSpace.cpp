// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/ShapeFunction.h>
#include <dolfin/Product.h>
#include <dolfin/ElementFunction.h>
#include <dolfin/Mapping.h>
#include <dolfin/FunctionSpace.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(int dim)
{
  if ( dim <= 0 )
    dolfin_error("Dimension of function space must be positive");
  
  _dim = dim;
  
  // Initialise the list of shape functions
  v.init(_dim);
  v.reset();
}
//-----------------------------------------------------------------------------
FunctionSpace::~FunctionSpace()
{

}
//-----------------------------------------------------------------------------
void FunctionSpace::add(ShapeFunction v)
{
  add(v, 0.0, 0.0, 0.0, 0.0);
}
//-----------------------------------------------------------------------------
void FunctionSpace::add(ShapeFunction v, ElementFunction dx)
{
  add(v, dx, 0.0, 0.0, 0.0);
}
//-----------------------------------------------------------------------------
void FunctionSpace::add(ShapeFunction v, ElementFunction dx, ElementFunction dy)
{
  add(v, dx, dy, 0.0, 0.0);
}
//-----------------------------------------------------------------------------
void FunctionSpace::add(ShapeFunction v,
			ElementFunction dx, ElementFunction dy, ElementFunction dz)
{
  add(v, dx, dy, dz, 0.0);
}
//-----------------------------------------------------------------------------
void FunctionSpace::add(ShapeFunction v,
			ElementFunction dx,
			ElementFunction dy,
			ElementFunction dz,
			ElementFunction dt)
{
  // Set derivatives of shape function
  v.set(dx, dy, dz, dt);
  
  // Add shape function
  if ( this->v.add(v) == -1 )
    dolfin_error("Function space is full.");
}
//-----------------------------------------------------------------------------
void FunctionSpace::add(ShapeFunction v, real dx)
{
  add(v, ElementFunction(dx));
}
//-----------------------------------------------------------------------------
void FunctionSpace::add(ShapeFunction v, real dx, real dy)
{
  add(v, ElementFunction(dx), ElementFunction(dy));
}
//-----------------------------------------------------------------------------
void FunctionSpace::add(ShapeFunction v, real dx, real dy, real dz)
{
  add(v, ElementFunction(dx), ElementFunction(dy), ElementFunction(dz));
}
//-----------------------------------------------------------------------------
void FunctionSpace::add(ShapeFunction v, real dx, real dy, real dz, real dt)
{
  add(v,
      ElementFunction(dx),
      ElementFunction(dy),
      ElementFunction(dz),
      ElementFunction(dt));
}
//-----------------------------------------------------------------------------
int FunctionSpace::dim() const
{
  return _dim;
}
//-----------------------------------------------------------------------------
void FunctionSpace::update(const Mapping& mapping)
{
  for (Iterator v(*this); !v.end(); ++v)
    v->update(mapping);
}
//-----------------------------------------------------------------------------
// FunctionSpace::Iterator
//-----------------------------------------------------------------------------
FunctionSpace::Iterator::Iterator
(const FunctionSpace &functionSpace) : V(functionSpace)
{
  v = V.v.begin();
}
//-----------------------------------------------------------------------------
int FunctionSpace::Iterator::dof(const Cell& cell) const
{
  return V.dof(v.index(), cell);
}
//-----------------------------------------------------------------------------
real FunctionSpace::Iterator::dof
(const Cell& cell, const ExpressionFunction& f, real t) const
{
  return V.dof(v.index(), cell, f, t);
}
//-----------------------------------------------------------------------------
int FunctionSpace::Iterator::index() const
{
  return v.index();
}
//-----------------------------------------------------------------------------
bool FunctionSpace::Iterator::end() const
{
  return v.end();
}
//-----------------------------------------------------------------------------
void FunctionSpace::Iterator::operator++()
{
  ++v;
}
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction* FunctionSpace::Iterator::pointer() const
{
  return &(*v);
}
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction& FunctionSpace::Iterator::operator*() const
{
  return *v;
}
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction* FunctionSpace::Iterator::operator->() const
{
  return &(*v);
}
//-----------------------------------------------------------------------------
