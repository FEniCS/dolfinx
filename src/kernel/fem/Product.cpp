#include <dolfin/FunctionList.h>
#include <dolfin/ShapeFunction.h>
#include <dolfin/ElementFunction.h>
#include <dolfin/Product.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionSpace::Product::Product()
{
  n = 1;
  _id = new int[1];
  _id[0] = 1; // Initialise to the function one
}
//-----------------------------------------------------------------------------
FunctionSpace::Product::Product(const ShapeFunction &v)
{
  n = 1;
  _id = new int[1];
  _id[0] = v.id();
}
//-----------------------------------------------------------------------------
FunctionSpace::Product::Product(const Product &v)
{
  n = v.n;
  
  _id = new int[n];

  for (int i = 0; i < n; i++)
	 _id[i] = v._id[i];
}
//-----------------------------------------------------------------------------
FunctionSpace::Product::Product(const ShapeFunction &v0,
										  const ShapeFunction &v1)
{
  if ( v0.one() ) {      // Special case: v0 = 1
	 n = 1;
	 _id = new int[1];
	 _id[0] = v1.id();
  }
  else if ( v1.one() ) { // Special case: v1 = 1
	 n = 1;
	 _id = new int[1];
	 _id[0] = v0.id();
  }
  else {
	 n = 2;
    _id = new int[2];
	 _id[0] = v0.id();
	 _id[1] = v1.id();
  }
}
//-----------------------------------------------------------------------------
FunctionSpace::Product::Product(const Product &v0, const Product &v1)
{
  if ( v0.one() ) {      // Special case: v0 = 1
	 n = v1.n;
	 _id = new int[n];
	 for (int i = 0; i < n; i++)
		_id[i] = v1._id[i];
  }
  else if ( v1.one() ) { // Special case: v1 = 1
	 n = v0.n;
	 _id = new int[n];
	 for (int i = 0; i < n; i++)
		_id[i] = v0._id[i];
  }
  else {
	 n = v0.n + v1.n;
	 
	 _id = new int[n];
	 
	 for (int i = 0; i < v0.n; i++)
		_id[i] = v0._id[i];
	 
	 for (int i = 0; i < v1.n; i++)
		_id[v0.n + i] = v1._id[i];
  }
}
//-----------------------------------------------------------------------------
FunctionSpace::Product::Product(const ShapeFunction &v0, const Product &v1)
{
  n = 1 + v1.n;
  
  _id = new int[n];
  
  _id[0] = v0.id();
  for (int i = 0; i < v1.n; i++)
	 _id[1 + i] = v1._id[i];
}
//-----------------------------------------------------------------------------
FunctionSpace::Product::~Product()
{
  if ( n > 0 )
	 delete [] _id;
}
//-----------------------------------------------------------------------------
void FunctionSpace::Product::set(const ShapeFunction &v0,
											const ShapeFunction &v1)
{
  if ( n > 0 )
	 delete [] _id;

  if ( v0.one() ) {      // Special case: v0 = 1
	 n = 1;
	 _id = new int[1];
	 _id[0] = v1.id();
  }
  else if ( v1.one() ) { // Special case: v1 = 1
	 n = 1;
	 _id = new int[1];
	 _id[0] = v0.id();
  }
  else {
	 n = 2;
	 _id = new int[2];
    _id[0] = v0.id();
	 _id[1] = v1.id();
  }
}
//-----------------------------------------------------------------------------
void FunctionSpace::Product::set(const Product &v0, const Product &v1)
{
  if ( n > 0 )
	 delete [] _id;

  if ( v0.one() ) {      // Special case: v0 = 1
	 n = v1.n;
	 _id = new int[n];
	 for (int i = 0; i < n; i++)
		_id[i] = v1._id[i];
  }
  else if ( v1.one() ) { // Special case: v1 = 1
	 n = v0.n;
	 _id = new int[n];
	 for (int i = 0; i < n; i++)
		_id[i] = v0._id[i];
  }
  else {
	 n = v0.n + v1.n;
	 _id = new int[n];
	 for (int i = 0; i < v0.n; i++)
		_id[i] = v0._id[i];
	 for (int i = 0; i < v1.n; i++)
		_id[v0.n + i] = v1._id[i];
  }
}
//-----------------------------------------------------------------------------
void FunctionSpace::Product::set(const ShapeFunction &v0, const Product &v1)
{
  if ( n > 0 )
	 delete [] _id;

  if ( v0.one() ) {      // Special case: v0 = 1
	 n = v1.n;
	 _id = new int[n];
	 for (int i = 0; i < n; i++)
		_id[i] = v1._id[i];
  }
  else if ( v1.one() ) { // Special case: v1 = 1
	 n = 1;
	 _id = new int[1];
	 _id[0] = v0.id();
  }
  else {
	 n = 1 + v1.n;
	 _id = new int[n];
	 _id[0] = v0.id();
	 for (int i = 0; i < v1.n; i++)
		_id[1 + i] = v1._id[i];
  }
}
//-----------------------------------------------------------------------------
int* FunctionSpace::Product::id() const
{
  return _id;
}
//-----------------------------------------------------------------------------
bool FunctionSpace::Product::zero() const
{
  return n == 1 && _id[0] == 0;
}
//-----------------------------------------------------------------------------
bool FunctionSpace::Product::one() const
{
  return n == 1 && _id[0] == 1;
}
//-----------------------------------------------------------------------------
int FunctionSpace::Product::size() const
{
  return n;
}
//-----------------------------------------------------------------------------
real FunctionSpace::Product::operator() (real x, real y, real z, real t) const
{
  real value = 1.0;

  for (int i = 0; i < n; i++)
	 value *= FunctionList::eval(_id[i], x, y, z, t);
  
  return value;
}
//-----------------------------------------------------------------------------
real FunctionSpace::Product::operator() (Point p) const
{
  real value = 1.0;
  
  for (int i = 0; i < n; i++)
	 value *= FunctionList::eval(_id[i], p.x, p.y, p.z, 0.0);

  return value;
}
//-----------------------------------------------------------------------------
FunctionSpace::Product&
FunctionSpace::Product::operator= (const ShapeFunction &v)
{
  if ( n != 1 ) {
	 delete [] _id;
	 n = 1;
	 _id = new int[1];
  }
  
  _id[0] = v.id();

  return *this;
}
//-----------------------------------------------------------------------------
FunctionSpace::Product&
FunctionSpace::Product::operator= (const Product &v)
{
  if ( n != v.n ) {
	 delete [] _id;
	 n = v.n;
	 _id = new int[n];
  }
	 
  for (int i = 0; i < n; i++)
	 _id[i] = v._id[i];

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
FunctionSpace::ElementFunction
FunctionSpace::Product::operator* (real a) const
{
  ElementFunction w(a, *this);
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
real FunctionSpace::Product::operator* (Integral::Measure &dm) const
{
  return dm * (*this);
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction dolfin::operator*
(real a, const FunctionSpace::Product &v)
{
  return v * a;
}
//-----------------------------------------------------------------------------
// Additional operators
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream,
												  const FunctionSpace::Product &v)
{
  stream << "[ Product with " << v.n << " factors: ";
  
  for (int  i = 0; i < v.n; i++)
	 stream << v._id[i] << " ";
  
  stream << "]";
  
  return stream;
}
//-----------------------------------------------------------------------------
