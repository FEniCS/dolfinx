// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Function.h>
#include <dolfin/ShapeFunction.h>
#include <dolfin/Product.h>
#include <dolfin/ElementFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction()
{
  n = 0;
  
  a = 0;
  v = 0;

  c = 0.0;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction(real a)
{
  n = 0;

  this->a = 0;
  v = 0;

  c = a;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction(const ShapeFunction& v)
{
  n = 1;

  a = new real[1];
  this->v = new Product[1];
  this->v[0] = v;

  a[0] = 1.0;

  c = 0.0;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction(const Product& v)
{
  n = 1;

  a = new real[1];
  this->v = new Product[1];
  this->v[0] = v;

  a[0] = 1.0;

  c = 0.0;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction(const ElementFunction& v)
{
  n = v.n;
  
  if ( n > 0 ) {
    a = new real[n];
    this->v = new Product[n];
    
    for (int i = 0; i < n; i++) {
      a[i] = v.a[i];
      this->v[i] = v.v[i];
    }
  }
  else {
    a = 0;
    this->v = 0;
  }
  
  c = v.c;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction(real a, const ShapeFunction& v)
{
  n = 1;
  
  this->a = new real[1];
  this->v = new Product[1];
  this->v[0] = v;

  this->a[0] = a;

  c = 0.0;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction(real a, const Product& v)
{
  n = 1;

  this->a = new real[1];
  this->v = new Product[1];
  this->v[0] = v;

  this->a[0] = a;

  c = 0.0;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction(real a,
						const ElementFunction& v)
{
  n = v.n;
  
  if ( n > 0 ) {
    this->a = new real[n];
    this->v = new Product[n];
    
    for (int i = 0; i < n; i++) {
      this->a[i] = a * v.a[i];
      this->v[i] = v.v[i];
    }
  }
  else {
    this->a = 0;
    this->v = 0;
  }
  
  c = a * v.c;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction
(real a0, const ShapeFunction& v0, real a1, const ShapeFunction& v1)
{
  n = 2;
  
  a = new real[2];
  a[0] = a0;
  a[1] = a1;
  
  v = new Product[2];
  v[0] = v0;
  v[1] = v1;

  c = 0.0;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction
(real a0, const Product& v0, real a1, const Product& v1)
{
  n = 2;

  a = new real[2];
  a[0] = a0;
  a[1] = a1;

  v = new Product[2];
  v[0] = v0;
  v[1] = v1;

  c = 0.0;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction
(real a0, const ElementFunction& v0, real a1, const ElementFunction& v1)
{
  n = v0.n + v1.n;

  if ( n > 0 ) {
    a = new real[n];
    v = new Product[n];
    
    for (int i = 0; i < v0.n; i++) {
      a[i] = a0 * v0.a[i];
      v[i] = v0.v[i];
    }
    
    for (int i = 0; i < v1.n; i++) {
      a[v0.n + i] = a1 * v1.a[i];
      v[v0.n + i] = v1.v[i];
    }
  }
  else {
    a = 0;
    v = 0;
  }
    
  c = a0*v0.c + a1*v1.c;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction
(real a0, const ShapeFunction& v0, real a1, const Product& v1)
{
  n = 2;
  
  a = new real[2];
  a[0] = a0;
  a[1] = a1;

  v = new Product[2];
  v[0] = v0;
  v[1] = v1;

  c = 0.0;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction
(real a0, const ShapeFunction& v0, real a1, const ElementFunction& v1)
{
  n = 1 + v1.n;

  a = new real[n];
  v = new Product[n];
  
  a[0] = a0;
  v[0] = v0;
  
  for (int i = 0; i < v1.n; i++) {
    a[1 + i] = a1 * v1.a[i];
    v[1 + i] = v1.v[i];
  }
  
  c = a1 * v1.c;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction
(real a0, const Product& v0, real a1, const ElementFunction& v1)
{
  n = 1 + v1.n;
  
  a = new real[n];
  v = new Product[n];
  
  a[0] = a0;
  v[0] = v0;
  
  for (int i = 0; i < v1.n; i++) {
    a[1 + i] = a1 * v1.a[i];
    v[1 + i] = v1.v[i];
  }
  
  c = a1 * v1.c;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction(const ShapeFunction& v0,
						const ShapeFunction& v1)
{
  n = 1;
  
  a = new real[1];
  v = new Product[1];
  v[0].set(v0, v1);
  
  a[0] = 1.0;

  c = 0.0;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction(const Product& v0,
						const Product& v1)
{
  n = 1;
  
  a = new real[1];
  v = new Product[1];
  v[0].set(v0, v1);
  
  a[0] = 1.0;

  c = 0.0;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction(const ElementFunction& v0,
						const ElementFunction& v1)
{
  n = v0.n*v1.n + v0.n + v1.n;
  
  // Check if constant
  if ( n == 0 ) {
    a = 0;
    v = 0;
    c = v0.c * v1.c;
    return;
  }
  
  a = new real[n];
  v = new Product[n];
  
  for (int i = 0; i < v0.n; i++)
    for (int j = 0; j < v1.n; j++) {
      a[i*v0.n + j] = v0.a[i] * v1.a[j];
      v[i*v0.n + j].set(v0.v[i],v1.v[j]);
    }
  int offset = v0.n * v1.n;
  
  for (int i = 0; i < v0.n; i++) {
    a[offset + i] = v1.c * v0.a[i];
    v[offset + i] = v0.v[i];
  }
  offset += v0.n;
  
  for (int i = 0; i < v1.n; i++) {
    a[offset + i] = v0.c * v1.a[i];
    v[offset + i] = v1.v[i];
  }
  
  c = v0.c * v1.c;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction(const ShapeFunction& v0,
						const Product& v1)
{
  n = 1;
  
  a = new real[1];
  v = new Product[1];
  v[0].set(v0, v1);
  
  a[0] = 1.0;
  
  c = 0.0;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction(const ShapeFunction&   v0,
						const ElementFunction& v1)
{
  n = v1.n + 1;
  
  a = new real[n];
  v = new Product[n];
  
  for (int i = 0; i < v1.n; i++) {
    a[i] = v1.a[i];
    v[i].set(v0, v1.v[i]); 
  }

  a[v1.n] = v1.c;
  v[v1.n] = v0;

  c = 0.0;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction(const Product&         v0,
						const ElementFunction& v1)
{
  n = v1.n + 1;
  
  a = new real[n];
  v = new Product[n];
  
  for (int i = 0; i < v1.n; i++) {
    a[i] = v1.a[i];
    v[i].set(v0, v1.v[i]);
  }

  a[v1.n] = v1.c;
  v[v1.n] = v0;
  
  c = 0.0;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::ElementFunction(const ElementFunction& v0,
						const ElementFunction& v1,
						const ElementFunction& v2,
						const ElementFunction& w0,
						const ElementFunction& w1,
						const ElementFunction& w2)
{
  // Computation of scalar product

  // Number of terms from sum * sum
  n = v0.n*w0.n + v1.n*w1.n + v2.n*w2.n;

  // Terms from constant * sum
  n += v0.n + v1.n + v2.n + w0.n + w1.n + w2.n;

  // Compute the constant
  c = v0.c*w0.c + v1.c*w1.c + v2.c*w2.c;

  // Check if all components are constant
  if ( n == 0 ) {
    a = 0;
    v = 0;
    return;
  }

  a = new real[n];
  v = new Product[n];

  // First term
  for (int i = 0; i < v0.n; i++)
    for (int j = 0; j < w0.n; j++) {
      a[i*v0.n + j] = v0.a[i] * w0.a[j];
      v[i*v0.n + j].set(v0.v[i],w0.v[j]);
    }
  int offset = v0.n * w0.n;

  // Second term
  for (int i = 0; i < v1.n; i++)
    for (int j = 0; j < w1.n; j++) {
      a[offset + i*v1.n + j] = v1.a[i] * w1.a[j];
      v[offset + i*v1.n + j].set(v1.v[i],w1.v[j]);
    }
  offset += v1.n * w1.n;
  
  // Third term
  for (int i = 0; i < v2.n; i++)
    for (int j = 0; j < w2.n; j++) {
      a[offset + i*v2.n + j] = v2.a[i] * w2.a[j];
      v[offset + i*v2.n + j].set(v2.v[i],w2.v[j]);
    }
  offset += v2.n * w2.n;

  //----------------------------------------------
  
  // Part 1 of first term from constant * sum
  for (int i = 0; i < v0.n; i++) {
    a[offset + i] = w0.c * v0.a[i];
    v[offset + i] = v0.v[i];
  }
  offset += v0.n;

  // Part 2 of first term from constant * sum
  for (int i = 0; i < w0.n; i++) {
    a[offset + i] = v0.c * w0.a[i];
    v[offset + i] = w0.v[i];
  }
  offset += w0.n;

  // Part 1 of second term from constant * sum
  for (int i = 0; i < v1.n; i++) {
    a[offset + i] = w1.c * v1.a[i];
    v[offset + i] = v1.v[i];
  }
  offset += v1.n;

  // Part 2 of second term from constant * sum
  for (int i = 0; i < w1.n; i++) {
    a[offset + i] = v1.c * w1.a[i];
    v[offset + i] = w1.v[i];
  }
  offset += w1.n;

  // Part 1 of third term from constant * sum
  for (int i = 0; i < v2.n; i++) {
    a[offset + i] = w2.c * v2.a[i];
    v[offset + i] = v2.v[i];
  }
  offset += v2.n;

  // Part 2 of third term from constant * sum
  for (int i = 0; i < w2.n; i++) {
    a[offset + i] = v2.c * w2.a[i];
    v[offset + i] = w2.v[i];
  }
  
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::~ElementFunction()
{
  if ( n > 0 ) {
    delete [] a;
    delete [] v;
  }
}
//-----------------------------------------------------------------------------
real FunctionSpace::ElementFunction::operator() 
(real x, real y, real z, real t) const
{
  real value = c;
  
  for (int i = 0; i < n; i++)
    value += a[i] * v[i](x,y,z,t);
  
  return value;
}
//-----------------------------------------------------------------------------
real FunctionSpace::ElementFunction::operator() (Point p) const
{
  real value = c;
  
  for (int i = 0; i < n; i++)
    value += a[i] * v[i](p);

  return value;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction& FunctionSpace::ElementFunction::operator=
(real a)
{
  if ( n > 0 ) {
    delete [] this->a;
    delete [] v;
  }
  
  n = 0;

  this->a = 0;
  v = 0;
  
  c = a;

  return *this;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction& FunctionSpace::ElementFunction::operator=
(const ShapeFunction& v)
{
  if ( n > 0 ) {
    delete [] a;
    delete [] this->v;
  }

  n = 1;

  a = new real[1];
  this->v = new Product[1];
  this->v[0] = v;

  a[0] = 1.0;

  c = 0;
  
  return *this;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction& FunctionSpace::ElementFunction::operator=
(const Product& v)
{
  if ( n > 0 ) {
    delete [] a;
    delete [] this->v;
  }

  n = 1;

  a = new real[1];
  this->v = new Product[1];
  this->v[0] = v;
  
  a[0] = 1.0;
  
  c = 0;
  
  return *this;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction& FunctionSpace::ElementFunction::operator=
(const ElementFunction& v)
{
  if ( n > 0 ) {
    delete [] a;
    delete [] this->v;
  }
  
  n = v.n;
  
  if ( n > 0 ) {
    a = new real[n];
    this->v = new Product[n];
    
    for (int i = 0; i < n; i++) {
      a[i] = v.a[i];
      this->v[i] = v.v[i];
    }
  }
  else {
    a = 0;
    this->v = 0;
  }

  c = v.c;
  
  return *this;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction FunctionSpace::ElementFunction::operator+
(const ShapeFunction& v) const
{
  ElementFunction w(1.0, v, 1.0, *this);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction FunctionSpace::ElementFunction::operator+
(const Product& v) const
{
  ElementFunction w(1.0, v, 1.0, *this);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction FunctionSpace::ElementFunction::operator+
(const ElementFunction& v) const
{
  ElementFunction w(1.0, v, 1.0, *this);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction FunctionSpace::ElementFunction::operator+=
(const ShapeFunction& v)
{
  real *new_a = new real[n + 1];
  Product *new_v = new Product[n + 1];

  for (int i = 0; i < n; i++) {
	 new_a[i] = a[i];
	 new_v[i] = this->v[i];
  }

  new_a[n] = 1.0;
  new_v[n] = v;

  delete [] a;
  delete [] this->v;
  
  a = new_a;
  this->v = new_v;
  n += 1;
  
  return *this;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction FunctionSpace::ElementFunction::operator+=
(const Product& v)
{
  real *new_a = new real[n + 1];
  Product *new_v = new Product[n + 1];

  for (int i = 0; i < n; i++) {
	 new_a[i] = a[i];
	 new_v[i] = this->v[i];
  }

  new_a[n] = 1.0;
  new_v[n] = v;

  delete [] a;
  delete [] this->v;
  
  a = new_a;
  this->v = new_v;
  n += 1;
  
  return *this;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction FunctionSpace::ElementFunction::operator+=
(const ElementFunction& v)
{
  real *new_a = new real[n + v.n];
  Product *new_v = new Product[n + v.n];

  for (int i = 0; i < n; i++) {
	 new_a[i] = a[i];
	 new_v[i] = this->v[i];
  }

  for (int i = 0; i < v.n; i++) {
	 new_a[n + i] = v.a[i];
	 new_v[n + i] = v.v[i];
  }

  delete [] a;
  delete [] this->v;
  
  a = new_a;
  this->v = new_v;
  n += v.n;

  c += v.c;
  
  return *this;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction FunctionSpace::ElementFunction::operator-
(const ShapeFunction& v) const
{
  ElementFunction w(-1.0, v, 1.0, *this);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction FunctionSpace::ElementFunction::operator-
(const Product& v) const
{
  ElementFunction w(-1.0, v, 1.0, *this);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction FunctionSpace::ElementFunction::operator-
(const ElementFunction& v) const
{
  ElementFunction w(1.0, *this, -1.0, v);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction FunctionSpace::ElementFunction::operator*
(real a) const
{
  ElementFunction w(a, *this);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction FunctionSpace::ElementFunction::operator*
(const ShapeFunction& v) const
{
  ElementFunction w(v, *this);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction FunctionSpace::ElementFunction::operator*
(const Product& v) const
{
  ElementFunction w(v, *this);
  return w;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction FunctionSpace::ElementFunction::operator*
(const ElementFunction& v) const
{
  ElementFunction w(v, *this);
  return w;
}
//-----------------------------------------------------------------------------
real FunctionSpace::ElementFunction::operator* (Integral::Measure& dm) const
{
  // The integral is linear
  real sum = c * dm;
  for (int i = 0; i < n; i++)
    sum += a[i] * ( dm * v[i] );
  
  return sum;
}
//-----------------------------------------------------------------------------
void FunctionSpace::ElementFunction::init(int size)
{
  if ( n == size )
	 return;

  n = size;
  
  delete [] a;
  delete [] v;

  a = new real[n];
  v = new Product[n]();

  for (int i = 0; i < n; i++)
    a[i] = 0.0;
}
//-----------------------------------------------------------------------------
void FunctionSpace::ElementFunction::set(int i, const ShapeFunction& v,
					 real value)
{
  a[i] = value;
  this->v[i] = v;
}
//-----------------------------------------------------------------------------
// Vector element function
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::Vector::Vector(int size)
{
  v = new ElementFunction[size];
  _size = size;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::Vector::Vector(const Vector& v)
{
  _size = v._size;
  this->v = new ElementFunction[_size];
  for (int i = 0; i < _size; i++)
    this->v[i] = v.v[i];
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::Vector::Vector(const ElementFunction& v0,
					       const ElementFunction& v1,
					       const ElementFunction& v2)
{
  _size = 3;
  v = new ElementFunction[_size];

  v[0] = v0;
  v[1] = v1;
  v[2] = v2;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction::Vector::~Vector()
{
  delete [] v;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction&
FunctionSpace::ElementFunction::Vector::operator() (int i)
{
  return v[i];
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction
FunctionSpace::ElementFunction::Vector::operator, (const Vector& v) const
{
  if ( _size != 3 || v._size != 3 )
    dolfin_error("Vector dimension must be 3 for scalar product.");
  
  ElementFunction w(this->v[0], this->v[1], this->v[2],
		    v.v[0], v.v[1], v.v[2]);

  return w;
}
//-----------------------------------------------------------------------------
int FunctionSpace::ElementFunction::Vector::size() const
{
  return _size;
}
//-----------------------------------------------------------------------------
// Additional operators
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction dolfin::operator*
(real a, const FunctionSpace::ElementFunction& v)
{
  return v * a;
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream,
				      const FunctionSpace::ElementFunction& v)
{
  stream << "[ ElementFunction with " << v.n
	 << " terms and offset = " << v.c << " ]";
  
  for (int  i = 0; i < v.n; i++) {
    if ( i == 0 )
      stream << endl;
    stream << "  " << v.a[i] << " * " << v.v[i];
    if ( i < (v.n - 1) )
      stream << endl;
  }
  
  return stream;
}
//-----------------------------------------------------------------------------
