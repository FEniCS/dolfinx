// (c) 2002 Johan Hoffman & Anders Logg, Chalmers Finite Element Center.
// Licensed under the GNU GPL Version 2.
//
// - Updates by Georgios Foufas (2002)
// - Updates by Johan Jansson (2003)

#include <dolfin/dolfin_math.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/Vector.h>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
Vector::Vector() : Variable("x", "A vector")
{
  values = 0;
  n = 0;
}
//-----------------------------------------------------------------------------
Vector::Vector(int size) : Variable("x", "A vector")
{
  if ( size < 0 )
    dolfin_error("Size of vector must be non-negative.");
  
  values = 0;
  n = 0;
  
  init(static_cast<unsigned int>(size));
}
//-----------------------------------------------------------------------------
Vector::Vector(unsigned int size) : Variable("x", "A vector")
{
  values = 0;
  n = 0;
  
  init(size);
}
//-----------------------------------------------------------------------------
Vector::Vector(const Vector& x) : Variable(x.name(), x.label())
{
  values = 0;
  n = 0;

  init(x.size());

  for (unsigned int i = 0; i < n; i++)
    values[i] = x.values[i];
}
//-----------------------------------------------------------------------------
Vector::Vector(real x0) : Variable("x", "A vector")
{
  values = 0;
  n = 0;
  
  init(1);
  values[0] = x0;
}
//-----------------------------------------------------------------------------
Vector::Vector(real x0, real x1) : Variable("x", "A vector")
{
  values = 0;
  n = 0;
  
  init(2);
  values[0] = x0;
  values[1] = x1;
}
//-----------------------------------------------------------------------------
Vector::Vector(real x0, real x1, real x2) : Variable("x", "A vector")
{
  values = 0;
  n = 0;
  
  init(3);
  values[0] = x0;
  values[1] = x1;
  values[2] = x2;
}
//-----------------------------------------------------------------------------
Vector::~Vector()
{
  clear();
}
//-----------------------------------------------------------------------------
void Vector::init(unsigned int size)
{
  // Two cases:
  //
  //   1. Already allocated and dimension changes -> reallocate
  //   2. Not allocated -> allocate
  //
  // Otherwise do nothing
  
  if ( values ) {
    if ( n != size ) {
      clear();      
      alloc(size);
    }
  }
  else
    alloc(size);
}
//-----------------------------------------------------------------------------
void Vector::clear()
{
  if ( values )
    delete [] values;
  values = 0;

  n = 0;
}
//-----------------------------------------------------------------------------
unsigned int Vector::size() const
{
  return n;
}
//-----------------------------------------------------------------------------
unsigned int Vector::bytes() const
{
  return sizeof(Vector) + n*sizeof(real);
}
//-----------------------------------------------------------------------------
real Vector::operator()(unsigned int i) const
{
  return values[i];
}
//-----------------------------------------------------------------------------
real& Vector::operator()(unsigned int i)
{
  return values[i];
}
//-----------------------------------------------------------------------------
void Vector::operator=(const Vector& x)
{
  init(x.size());

  for (unsigned int i = 0; i < n; i++)
    values[i] = x.values[i];    
}
//-----------------------------------------------------------------------------
void Vector::operator=(real a)
{
  for (unsigned int i = 0; i < n; i++)
    values[i] = a;    
}
//-----------------------------------------------------------------------------
void Vector::operator=(const Matrix::Row& row)
{
  if ( n != row.size() )
    dolfin_error("Matrix dimensions don't match.");

  for (unsigned int i = 0; i < n; i++)
    values[i] = row(i);
}
//-----------------------------------------------------------------------------
void Vector::operator=(const Matrix::Column& col)
{
  if ( n != col.size() )
    dolfin_error("Matrix dimensions don't match.");

  for (unsigned int i = 0; i < n; i++)
    values[i] = col(i);
}
//-----------------------------------------------------------------------------
void Vector::operator+=(const Vector& x)
{
  if ( n != x.size() )
    dolfin_error("Matrix dimensions don't match.");

  for (unsigned int i = 0; i < n; i++)
    values[i] += x.values[i];
}
//-----------------------------------------------------------------------------
void Vector::operator+=(const Matrix::Row& row)
{
  if ( n != row.size() )
    dolfin_error("Matrix dimensions don't match.");
                                                                                                                                                            
  for (unsigned int i = 0; i < n; i++)
    values[i] += row(i);
}
//-----------------------------------------------------------------------------
void Vector::operator+=(const Matrix::Column& col)
{
  if ( n != col.size() )
    dolfin_error("Matrix dimensions don't match.");
                                                                                                                                                            
  for (unsigned int i = 0; i < n; i++)
    values[i] += col(i);
}
//-----------------------------------------------------------------------------
void Vector::operator+=(real a) 
{
  for (unsigned int i = 0; i < n; i++)
    values[i] += a;
}
//-----------------------------------------------------------------------------
void Vector::operator-=(const Vector& x)
{
  for (unsigned int i = 0; i < n; i++)
    values[i] -= x.values[i];
}
//-----------------------------------------------------------------------------
void Vector::operator-=(const Matrix::Row& row)
{
  if ( n != row.size() )
    dolfin_error("Matrix dimensions don't match.");
                                                                                                                                                            
  for (unsigned int i = 0; i < n; i++)
    values[i] -= row(i);
}
//-----------------------------------------------------------------------------
void Vector::operator-=(const Matrix::Column& col)
{
  if ( n != col.size() )
    dolfin_error("Matrix dimensions don't match.");
                                                                                                                                                            
  for (unsigned int i = 0; i < n; i++)
    values[i] -= col(i);
}
//-----------------------------------------------------------------------------
void Vector::operator-=(real a) 
{
  for (unsigned int i = 0; i < n; i++)
    values[i] -= a;
}
//-----------------------------------------------------------------------------
void Vector::operator*=(real a)
{
  for (unsigned int i = 0; i < n; i++)
    values[i] *= a; 
}
//-----------------------------------------------------------------------------
real Vector::operator*(const Vector& x) const
{
  if ( n != x.n )
    dolfin_error("Matrix dimensions don't match.");

  real sum = 0.0;
  for (unsigned int i = 0; i < n; i++)
    sum += values[i] * x.values[i];
  
  return sum;
}
//-----------------------------------------------------------------------------
real Vector::operator*(const Matrix::Row& row) const
{
  return row * (*this);
}
//-----------------------------------------------------------------------------
real Vector::operator*(const Matrix::Column& col) const
{
  return col * (*this);
}
//-----------------------------------------------------------------------------
real Vector::norm() const
{
  return norm(2);
}
//-----------------------------------------------------------------------------
real Vector::norm(unsigned int i) const
{
  real norm = 0.0; 

  switch(i){
  case 0:
    // max-norm
    for (unsigned int i = 0; i < n; i++)
      if ( fabs(values[i]) > norm )
	norm = fabs(values[i]);
    return norm;
    break;
  case 1:
    // l1-norm
    for (unsigned int i = 0; i < n; i++)
      norm += fabs(values[i]);
    return norm;
    break;
  case 2:
    // l2-norm
    for (unsigned int i = 0; i < n; i++)
      norm += values[i] * values[i];
    return sqrt(norm);
    break;
  default:
    cout << "Unknown vector norm" << endl;
    exit(1);
  }  

}
//-----------------------------------------------------------------------------
void Vector::add(real a, Vector& x)
{
  if ( n != x.n )
    dolfin_error("Matrix dimensions don't match.");
  
  for (unsigned int i = 0; i < n; i++)
    values[i] += a * x.values[i];  
}
//-----------------------------------------------------------------------------
void Vector::add(real a, const Matrix::Row& row)
{
  if ( n != row.size() )
    dolfin_error("Matrix dimensions don't match.");
                                                                                                                                                            
  for (unsigned int i = 0; i < n; i++)
    values[i] += a * row(i);
}
//-----------------------------------------------------------------------------
void Vector::add(real a, const Matrix::Column& col)
{
  if ( n != col.size() )
    dolfin_error("Matrix dimensions don't match.");
                                                                                                                                                            
  for (unsigned int i = 0; i < n; i++)
    values[i] += a * col(i);
}
//-----------------------------------------------------------------------------
void Vector::rand()
{
  for (unsigned int i = 0; i < n; i++)
    values[i] = dolfin::rand();
}
//-----------------------------------------------------------------------------
void Vector::show() const
{
  cout << "[ ";
  for (unsigned int i = 0; i < n; i++)
    cout << values[i] << " ";
  cout << "]" << endl;
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream, const Vector& x)
{
  stream << "[ Vector of size " << x.size()
	 << ", approximatetly ";
  
  unsigned int bytes = x.bytes();
  
  if ( bytes > 1024*1024 )
    stream << bytes/1024 << " Mb. ]";
  else if ( bytes > 1024 )
    stream << bytes/1024 << " kb. ]";
  else
    stream << bytes << " bytes ]";
  
  return stream;
}
//-----------------------------------------------------------------------------
void Vector::alloc(unsigned int size)
{
  // Use with caution. Only for internal use.

  values = new real[size];
  for (unsigned int i = 0; i < size; i++)
    values[i] = 0.0;
  n = size;
}
//-----------------------------------------------------------------------------
