// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <iostream.h>

#include <dolfin/ShapeFunction.h>
#include <dolfin/Product.h>
#include <dolfin/ElementFunction.h>
#include <dolfin/FunctionSpace.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(int dim)
{
  // FIXME: Use logging system
  if ( dim <= 0 ) {
	 cout << "Error: dimension for function space must be positive" << endl;
	 exit(1);
  }

  this->dim = dim;
  current = 0;
  
  // Initialise the list of shape functions
  v  = new ShapeFunction[dim];
}
//-----------------------------------------------------------------------------
FunctionSpace::~FunctionSpace()
{
  if ( v )
	 delete [] v;
  v = 0;
}
//-----------------------------------------------------------------------------
int FunctionSpace::add(ShapeFunction v)
{
  add(v, 0.0, 0.0, 0.0, 0.0);
}
//-----------------------------------------------------------------------------
int FunctionSpace::add(ShapeFunction v, ElementFunction dx)
{
  add(v, dx, 0.0, 0.0, 0.0);
}
//-----------------------------------------------------------------------------
int FunctionSpace::add(ShapeFunction v, ElementFunction dx, ElementFunction dy)
{
  add(v, dx, dy, 0.0, 0.0);
}
//-----------------------------------------------------------------------------
int FunctionSpace::add(ShapeFunction v,
							  ElementFunction dx, ElementFunction dy, ElementFunction dz)
{
  add(v, dx, dy, dz, 0.0);
}
//-----------------------------------------------------------------------------
int FunctionSpace::add(ShapeFunction v,
							  ElementFunction dx,
							  ElementFunction dy,
							  ElementFunction dz,
							  ElementFunction dt)
{
  // FIXME: Use loggin system
  if ( current >= dim ) {
	 cout << "Error: function space is full." << endl;
	 exit(1);
  }

  v.set(dx, dy, dz, dt);
  this->v[current] = v;

  current++;
}
//-----------------------------------------------------------------------------
