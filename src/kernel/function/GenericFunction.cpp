// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Mesh.h>
#include <dolfin/GenericFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericFunction::GenericFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
GenericFunction::~GenericFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real GenericFunction::operator() (const Node&  n, real t) const
{
  dolfin_error("Function can not be evaluated at a given node.");
  return 0.0;
}
//-----------------------------------------------------------------------------
real GenericFunction::operator() (const Point& p, real t) const
{
  dolfin_error("Function can not be evaluated at a given point.");
  return 0.0;
}
//-----------------------------------------------------------------------------
real GenericFunction::operator() (real x, real y, real z, real t) const
{
  dolfin_error("Function can not be evaluated at given coordinates.");
  return 0.0;
}
//-----------------------------------------------------------------------------
real GenericFunction::operator() (unsigned int i, real t) const
{
  dolfin_error("Function can not be evaluated for given a component.");
  return 0.0;
}
//-----------------------------------------------------------------------------
void GenericFunction::update(real t)
{
  dolfin_error("Function can not be updated to given time.");
}
//-----------------------------------------------------------------------------
real GenericFunction::time() const
{
  dolfin_error("Function is not specified at a given time.");
  return 0.0;
}
//-----------------------------------------------------------------------------
Mesh& GenericFunction::mesh() const
{
  dolfin_error("Function is not defined on a mesh.");
}
//-----------------------------------------------------------------------------
void GenericFunction::update(FunctionSpace::ElementFunction &v,
			     const FiniteElement &element,
			     const Cell &cell, real t) const
{
  dolfin_error("Function is not defined on a mesh.");
}
//-----------------------------------------------------------------------------
