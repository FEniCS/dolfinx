// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/ShapeFunction.h>
#include <dolfin/Product.h>
#include <dolfin/ElementFunction.h>
#include <dolfin/Map.h>
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
  // Do nothing
}
//-----------------------------------------------------------------------------
void FunctionSpace::add(ShapeFunction v)
{
  add(v, 0.0, 0.0, 0.0, 0.0);
}
//-----------------------------------------------------------------------------
void FunctionSpace::add(ShapeFunction v, ElementFunction ddx)
{
  add(v, ddx, 0.0, 0.0, 0.0);
}
//-----------------------------------------------------------------------------
void FunctionSpace::add(ShapeFunction v, ElementFunction ddx, ElementFunction ddy)
{
  add(v, ddx, ddy, 0.0, 0.0);
}
//-----------------------------------------------------------------------------
void FunctionSpace::add(ShapeFunction v,
			ElementFunction ddx, ElementFunction ddy, ElementFunction ddz)
{
  add(v, ddx, ddy, ddz, 0.0);
}
//-----------------------------------------------------------------------------
void FunctionSpace::add(ShapeFunction v,
			ElementFunction ddx,
			ElementFunction ddy,
			ElementFunction ddz,
			ElementFunction ddt)
{
  // Set derivatives of shape function
  v.set(ddx, ddy, ddz, ddt);
  
  // Add shape function
  if ( this->v.add(v) == -1 )
    dolfin_error("Function space is full.");
}
//-----------------------------------------------------------------------------
void FunctionSpace::add(ShapeFunction v, real ddx)
{
  add(v, ElementFunction(ddx));
}
//-----------------------------------------------------------------------------
void FunctionSpace::add(ShapeFunction v, real ddx, real ddy)
{
  add(v, ElementFunction(ddx), ElementFunction(ddy));
}
//-----------------------------------------------------------------------------
void FunctionSpace::add(ShapeFunction v, real ddx, real ddy, real ddz)
{
  add(v, ElementFunction(ddx), ElementFunction(ddy), ElementFunction(ddz));
}
//-----------------------------------------------------------------------------
void FunctionSpace::add(ShapeFunction v, real ddx, real ddy, real ddz, real ddt)
{
  add(v,
      ElementFunction(ddx),
      ElementFunction(ddy),
      ElementFunction(ddz),
      ElementFunction(ddt));
}
//-----------------------------------------------------------------------------
int FunctionSpace::dim() const
{
  return _dim;
}
//-----------------------------------------------------------------------------
void FunctionSpace::update(const Map& map)
{
  for (Iterator v(*this); !v.end(); ++v)
    v->update(map);
}
//-----------------------------------------------------------------------------
// FunctionSpace::Iterator
//-----------------------------------------------------------------------------
FunctionSpace::Iterator::Iterator
(const FunctionSpace &functionSpace) : V(&functionSpace)
{
  //dolfin_debug1("size: %d", V->v.size());

  v = V->v.begin();
}
//-----------------------------------------------------------------------------
int FunctionSpace::Iterator::dof(const Cell& cell) const
{
  return V->dof(v.index(), cell);
}
//-----------------------------------------------------------------------------
real FunctionSpace::Iterator::dof
(const Cell& cell, const ExpressionFunction& f, real t) const
{
  return V->dof(v.index(), cell, f, t);
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
// Vector function space
//-----------------------------------------------------------------------------
FunctionSpace::Vector::Vector(int size)
{
  v = new FunctionSpace*[size];
  _size = size;
}
//-----------------------------------------------------------------------------
FunctionSpace::Vector::Vector(const Vector& v)
{
  _size = v._size;
  this->v = new FunctionSpace*[_size];
  for (int i = 0; i < _size; i++)
    this->v[i] = v.v[i];
}
//-----------------------------------------------------------------------------
FunctionSpace::Vector::~Vector()
{
  delete [] v;
}
//-----------------------------------------------------------------------------
FunctionSpace&
FunctionSpace::Vector::operator() (int i)
{
  return *(v[i]);
}
//-----------------------------------------------------------------------------
int FunctionSpace::Vector::size() const
{
  return _size;
}
//-----------------------------------------------------------------------------
