// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Cell.h>
#include <dolfin/Mapping.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Mapping::Mapping()
{
  reset();
}
//-----------------------------------------------------------------------------
real Mapping::det() const
{
  return d;
}
//-----------------------------------------------------------------------------
real Mapping::dx(real a) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real Mapping::dy(real a) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real Mapping::dz(real a) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real Mapping::dt(real a) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction Mapping::dx
(const FunctionSpace::Product &v) const
{
  std::cout << "Warning: derivative of Product not implemented." << std::endl;
  
  FunctionSpace::ElementFunction w;
  return w;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction Mapping::dy
(const FunctionSpace::Product &v) const
{
  std::cout << "Warning: derivative of Product not implemented." << std::endl;
  
  FunctionSpace::ElementFunction w;
  return w;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction Mapping::dz
(const FunctionSpace::Product &v) const
{
  std::cout << "Warning: derivative of Product not implemented." << std::endl;
  
  FunctionSpace::ElementFunction w;
  return w;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction Mapping::dt
(const FunctionSpace::Product &v) const
{
  std::cout << "Warning: derivative of Product not implemented." << std::endl;
  
  FunctionSpace::ElementFunction w;
  return w;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction Mapping::dx
(const FunctionSpace::ElementFunction &v) const
{
  std::cout << "Warning: derivative of ElementFunction not implemented." << std::endl;
  
  FunctionSpace::ElementFunction w;
  return w;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction Mapping::dy
(const FunctionSpace::ElementFunction &v) const
{
  std::cout << "Warning: derivative of ElementFunction not implemented." << std::endl;
  
  FunctionSpace::ElementFunction w;
  return w;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction Mapping::dz
(const FunctionSpace::ElementFunction &v) const
{
  std::cout << "Warning: derivative of ElementFunction not implemented." << std::endl;
  
  FunctionSpace::ElementFunction w;
  return w;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction Mapping::dt
(const FunctionSpace::ElementFunction &v) const
{
  std::cout << "Warning: derivative of ElementFunction not implemented." << std::endl;
  
  FunctionSpace::ElementFunction w;
  return w;
}
//-----------------------------------------------------------------------------
void Mapping::reset()
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
