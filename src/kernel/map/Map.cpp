// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Cell.h>
#include <dolfin/Map.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Map::Map()
{
  reset();
}
//-----------------------------------------------------------------------------
real Map::det() const
{
  return d;
}
//-----------------------------------------------------------------------------
real Map::dx(real a) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real Map::dy(real a) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real Map::dz(real a) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real Map::dt(real a) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction Map::dx
(const FunctionSpace::Product &v) const
{
  dolfin_warning("Derivative of Product not implemented.");
  
  FunctionSpace::ElementFunction w;
  return w;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction Map::dy
(const FunctionSpace::Product &v) const
{
  dolfin_warning("Derivative of Product not implemented.");
  
  FunctionSpace::ElementFunction w;
  return w;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction Map::dz
(const FunctionSpace::Product &v) const
{
  dolfin_warning("Derivative of Product not implemented.");
  
  FunctionSpace::ElementFunction w;
  return w;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction Map::dt
(const FunctionSpace::Product &v) const
{
  dolfin_warning("Derivative of Product not implemented.");
  
  FunctionSpace::ElementFunction w;
  return w;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction Map::dx
(const FunctionSpace::ElementFunction &v) const
{
  dolfin_warning("Derivative of ElementFunction not implemented.");
  
  FunctionSpace::ElementFunction w;
  return w;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction Map::dy
(const FunctionSpace::ElementFunction &v) const
{
  dolfin_warning("Derivative of ElementFunction not implemented.");
  
  FunctionSpace::ElementFunction w;
  return w;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction Map::dz
(const FunctionSpace::ElementFunction &v) const
{
  dolfin_warning("Derivative of ElementFunction not implemented.");
  
  FunctionSpace::ElementFunction w;
  return w;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction Map::dt
(const FunctionSpace::ElementFunction &v) const
{
  dolfin_warning("Derivative of ElementFunction not implemented.");
  
  FunctionSpace::ElementFunction w;
  return w;
}
//-----------------------------------------------------------------------------
void Map::reset()
{
  f11 = f12 = f13 = 0.0;
  f21 = f22 = f23 = 0.0;
  f31 = f32 = f33 = 0.0;

  g11 = g12 = g13 = 0.0;
  g21 = g22 = g23 = 0.0;
  g31 = g32 = g33 = 0.0;
  
  d = 0.0;
}
//-----------------------------------------------------------------------------
