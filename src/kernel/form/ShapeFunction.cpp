// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/FunctionList.h>
#include <dolfin/Product.h>
#include <dolfin/ElementFunction.h>
#include <dolfin/ShapeFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction::ShapeFunction()
{
  _id = 0; // Initialise to the zero function
  _component = 0; // Initialise to the zero function
}
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction::ShapeFunction(int i, int component)
{
  _id = 1; // Initialise to one
  _component = component;
}
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction::ShapeFunction(function f)
{
  _id = FunctionList::add(f);
}
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction::ShapeFunction(const ShapeFunction &v)
{
  this->_id = v._id;
}
//-----------------------------------------------------------------------------
void FunctionSpace::ShapeFunction::set(ElementFunction ddx,
				       ElementFunction ddy,
				       ElementFunction ddz,
				       ElementFunction ddt)
{
  FunctionList::set(_id, ddx, ddy, ddz, ddt);
}
//-----------------------------------------------------------------------------
int FunctionSpace::ShapeFunction::id() const
{
  return _id;
}
//-----------------------------------------------------------------------------
int FunctionSpace::ShapeFunction::component() const
{
  return _component;
}
//-----------------------------------------------------------------------------
bool FunctionSpace::ShapeFunction::zero() const
{
  return _id == 0;
}
//-----------------------------------------------------------------------------
bool FunctionSpace::ShapeFunction::one() const
{
  return _id == 1;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionSpace::ShapeFunction::ddx() const
{
  return FunctionList::ddx(_id);
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionSpace::ShapeFunction::ddy() const
{
  return FunctionList::ddy(_id);
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionSpace::ShapeFunction::ddz() const
{
  return FunctionList::ddz(_id);
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionSpace::ShapeFunction::ddt() const
{
  return FunctionList::ddt(_id);
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionSpace::ShapeFunction::ddX() const
{
  return FunctionList::ddX(_id);
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionSpace::ShapeFunction::ddY() const
{
  return FunctionList::ddY(_id);
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionSpace::ShapeFunction::ddZ() const
{
  return FunctionList::ddZ(_id);
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionSpace::ShapeFunction::ddT() const
{
  return FunctionList::ddT(_id);
}
//-----------------------------------------------------------------------------
void FunctionSpace::ShapeFunction::update(const Map& map)
{
  FunctionList::update(*this, map);
}
//-----------------------------------------------------------------------------
real
FunctionSpace::ShapeFunction::operator() (real x, real y, real z, real t) const
{
  return FunctionList::eval(_id, x, y, z, t);
}
//-----------------------------------------------------------------------------
real FunctionSpace::ShapeFunction::operator() (Point p) const
{
  // Warning: value of t is ignored
  return FunctionList::eval(_id, p.x, p.y, p.z, 0.0);
}
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction&
FunctionSpace::ShapeFunction::operator= (const ShapeFunction &v)
{
  this->_id = v._id;

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
real FunctionSpace::ShapeFunction::operator* (Integral::Measure &dm) const
{
  return dm * (*this);
}
//-----------------------------------------------------------------------------
void FunctionSpace::ShapeFunction::operator= (int zero)
{
  if ( zero != 0 )
	 dolfin_error("Assignment to int must be zero.");

  _id = -1;
}
//-----------------------------------------------------------------------------
bool FunctionSpace::ShapeFunction::operator! () const
{
  return _id == -1;
}
//-----------------------------------------------------------------------------
// Additional operators
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction dolfin::operator*
(real a, const FunctionSpace::ShapeFunction &v)
{
  return v * a;
}
//-----------------------------------------------------------------------------
// Vector element function
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction::Vector::Vector(int size)
{
  v = new ShapeFunction[size];
  _size = size;
}
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction::Vector::Vector(const Vector& v)
{
  _size = v._size;
  this->v = new ShapeFunction[_size];
  for (int i = 0; i < _size; i++)
    this->v[i] = v.v[i];
}
//-----------------------------------------------------------------------------
/*
FunctionSpace::ShapeFunction::Vector::Vector(const ShapeFunction& v0,
					       const ShapeFunction& v1,
					       const ShapeFunction& v2)
{
  _size = 3;
  v = new ShapeFunction[_size];

  v[0] = v0;
  v[1] = v1;
  v[2] = v2;
}
*/
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction::Vector::~Vector()
{
  delete [] v;
}
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction&
FunctionSpace::ShapeFunction::Vector::operator() (int i)
{
  return v[i];
}
//-----------------------------------------------------------------------------
int FunctionSpace::ShapeFunction::Vector::size() const
{
  return _size;
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream,
				       const FunctionSpace::ShapeFunction &v)
{
  stream << "[ ShapeFunction with id = " << v._id << " ]";
  
  return stream;
}
//-----------------------------------------------------------------------------
