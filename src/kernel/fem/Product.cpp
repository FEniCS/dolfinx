#include <dolfin/ShapeFunction.h>
#include <dolfin/ElementFunction.h>
#include <dolfin/Product.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionSpace::Product::Product()
{
  n = 1;
  v = new ShapeFunction[n](1);
}
//-----------------------------------------------------------------------------
FunctionSpace::Product::Product(const ShapeFunction &v)
{
  n = 1;
  this->v = new ShapeFunction[n](v);
}
//-----------------------------------------------------------------------------
FunctionSpace::Product::Product(const Product &v)
{
  n = v.n;
  
  this->v = new ShapeFunction[n];

  for (int i = 0; i < n; i++)
	 this->v[i] = v.v[i];
}
//-----------------------------------------------------------------------------
FunctionSpace::Product::Product(const ShapeFunction &v0,
										  const ShapeFunction &v1)
{
  n = 2;
  
  v = new ShapeFunction[n];

  v[0] = v0;
  v[1] = v1;
}
//-----------------------------------------------------------------------------
FunctionSpace::Product::Product(const Product &v0, const Product &v1)
{
  n = v0.n + v1.n;

  v = new ShapeFunction[n];

  for (int i = 0; i < v0.n; i++)
	 v[i] = v0.v[i];

  for (int i = 0; i < v1.n; i++)
	 v[v0.n + i] = v1.v[i];
}
//-----------------------------------------------------------------------------
FunctionSpace::Product::Product(const ShapeFunction &v0, const Product &v1)
{
  n = 1 + v1.n;
  
  v = new ShapeFunction[n];
  
  v[0] = v0;
  for (int i = 0; i < v1.n; i++)
	 v[1 + i] = v1.v[i];
}
//-----------------------------------------------------------------------------
FunctionSpace::Product::~Product()
{
  if ( n > 0 )
	 delete [] v;
}
//-----------------------------------------------------------------------------
void FunctionSpace::Product::set(const ShapeFunction &v0,
											  const ShapeFunction &v1)
{
  if ( n > 0 )
	 delete [] v;

  n = 2;
  
  v = new ShapeFunction[n];
  
  v[0] = v0;
  v[1] = v1;
}
//-----------------------------------------------------------------------------
void FunctionSpace::Product::set(const Product &v0, const Product &v1)
{
  if ( n > 0 )
	 delete [] v;

  n = v0.n + v1.n;

  v = new ShapeFunction[n];

  for (int i = 0; i < v0.n; i++)
	 v[i] = v0.v[i];

  for (int i = 0; i < v1.n; i++)
	 v[v0.n + i] = v1.v[i];
}
//-----------------------------------------------------------------------------
void FunctionSpace::Product::set(const ShapeFunction &v0, const Product &v1)
{
  if ( n > 0 )
	 delete [] v;

  n = 1 + v1.n;

  v = new ShapeFunction[n];

  v[0] = v0;
  for (int i = 0; i < v1.n; i++)
	 v[1 + i] = v1.v[i];
}
//-----------------------------------------------------------------------------
real FunctionSpace::Product::operator() (real x, real y, real z, real t)
{
  real value = 1.0;

  for (int i = 0; i < n; i++)
	 value *= v[i](x,y,z,t);

  return value;
}
//-----------------------------------------------------------------------------
FunctionSpace::Product&
FunctionSpace::Product::operator= (const ShapeFunction &v)
{
  if ( n > 0 )
	 delete [] this->v;

  n = 1;

  this->v = new ShapeFunction[n](v);

  return *this;
}
//-----------------------------------------------------------------------------
FunctionSpace::Product&
FunctionSpace::Product::operator= (const Product &v)
{
  if ( n > 0 )
	 delete [] this->v;

  n = v.n;

  this->v = new ShapeFunction[n];

  for (int i = 0; i < n; i++)
	 this->v[i] = v.v[i];

  return *this;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction
FunctionSpace::Product::operator+ (const ShapeFunction &v) const
{
  ElementFunction w(1.0, v, 1.0, *this);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction
FunctionSpace::Product::operator+ (const Product &v) const
{
  ElementFunction w(1.0, v, 1.0, *this);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction
FunctionSpace::Product::operator+ (const ElementFunction &v)  const
{
  ElementFunction w(1.0, *this, 1.0, v);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction
FunctionSpace::Product::operator- (const ShapeFunction &v) const
{
  ElementFunction w(-1.0, v, 1.0, *this);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction
FunctionSpace::Product::operator- (const Product &v) const
{
  ElementFunction w(1.0, *this, -1.0, v);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction
FunctionSpace::Product::operator- (const ElementFunction &v) const
{
  ElementFunction w(1.0, *this, -1.0, v);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::Product
FunctionSpace::Product::operator* (const ShapeFunction &v) const
{
  Product w(v, *this);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::Product
FunctionSpace::Product::operator* (const Product &v) const
{
  Product w(v, *this);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction
FunctionSpace::Product::operator* (const ElementFunction &v) const
{
  ElementFunction w(*this, v);
  return w;
}
//-----------------------------------------------------------------------------
