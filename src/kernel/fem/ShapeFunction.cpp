#include <iostream>

#include <dolfin/FunctionList.h>
#include <dolfin/Product.h>
#include <dolfin/ElementFunction.h>
#include <dolfin/ShapeFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction::ShapeFunction()
{
  id = 0; // Initialise to the zero function
}
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction::ShapeFunction(int i)
{
  id = 1; // Initialise to one
}
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction::ShapeFunction(function f)
{
  id = FunctionList::add(f);
}
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction::ShapeFunction(const ShapeFunction &v)
{
  this->id = v.id;
}
//-----------------------------------------------------------------------------
void FunctionSpace::ShapeFunction::set(ElementFunction dx,
													ElementFunction dy,
													ElementFunction dz,
													ElementFunction dt)
{
  FunctionList::set(id, dx, dy, dz, dt);
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction FunctionSpace::ShapeFunction::dx() const
{
  return FunctionList::dx(id);
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction FunctionSpace::ShapeFunction::dy() const
{
  return FunctionList::dy(id);
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction FunctionSpace::ShapeFunction::dz() const
{
  return FunctionList::dz(id);
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction FunctionSpace::ShapeFunction::dt() const
{
  return FunctionList::dt(id);
}
//-----------------------------------------------------------------------------
real
FunctionSpace::ShapeFunction::operator() (real x, real y, real z, real t) const
{
  return FunctionList::eval(id,x,y,z,t);
}
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction&
FunctionSpace::ShapeFunction::operator= (const ShapeFunction &v)
{
  this->id = v.id;

  return *this;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction
FunctionSpace::ShapeFunction::operator+ (const ShapeFunction &v) const
{
  ElementFunction w(1.0, *this, 1.0, v);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction
FunctionSpace::ShapeFunction::operator+ (const FunctionSpace::Product &v) const
{
  ElementFunction w(1.0, *this, 1.0, v);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction
FunctionSpace::ShapeFunction::operator+ (const ElementFunction &v) const
{
  ElementFunction w(1.0, *this, 1.0, v);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction
FunctionSpace::ShapeFunction::operator- (const ShapeFunction &v) const
{
  ElementFunction w(1.0, *this, -1.0, v);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction
FunctionSpace::ShapeFunction::operator- (const Product &v) const
{
  ElementFunction w(1.0, *this, -1.0, v);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction
FunctionSpace::ShapeFunction::operator- (const ElementFunction &v) const
{
  ElementFunction w(1.0, *this, -1.0, v);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::Product
FunctionSpace::ShapeFunction::operator* (const ShapeFunction &v) const
{
  Product w(*this, v);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::Product
FunctionSpace::ShapeFunction::operator* (const Product &v) const
{
  Product w(*this, v);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction
FunctionSpace::ShapeFunction::operator* (const ElementFunction &v) const
{
  ElementFunction w(*this, v);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction
FunctionSpace::ShapeFunction::operator* (real a) const
{
  ElementFunction w(a, *this);
  return w;
}
//-----------------------------------------------------------------------------
