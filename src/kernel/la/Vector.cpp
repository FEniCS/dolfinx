// (c) 2002 Johan Hoffman & Anders Logg, Chalmers Finite Element Center.
// Licensed under the GNU GPL Version 2.
//
// Contributions by: Georgios Foufas (2002)
//                   Johan Jansson (2003)

#include <math.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/Vector.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Vector::Vector()
{
  n = 0;
  values = 0;

  rename("x", "A vector");
}
//-----------------------------------------------------------------------------
Vector::Vector(int size)
{
  values = 0;

  init(size);

  rename("x", "A vector");
}
//-----------------------------------------------------------------------------
Vector::Vector(const Vector &vector)
{
  n = vector.n;
  values = new real[n];
  for (int i = 0; i < n; i++)
	 values[i] = vector.values[i];

  rename(vector.name(), vector.label());
}
//-----------------------------------------------------------------------------
int Vector::bytes() const
{
  return sizeof(Vector) + n*sizeof(real);
}
//-----------------------------------------------------------------------------
Vector::~Vector()
{
  delete [] values;
}
//-----------------------------------------------------------------------------
void Vector::init(int size)
{
  if ( values )
	 delete [] values;

  n = size;

  values = new real[n];

  for (int i=0;i<n;i++)
	 values[i] = 0.0;
}
//-----------------------------------------------------------------------------
int Vector::size() const
{
  return n;
}
//-----------------------------------------------------------------------------
real& Vector::operator()(int i)
{
  return values[i];
}
//-----------------------------------------------------------------------------
real Vector::operator()(int i) const
{
  return values[i];
}
//-----------------------------------------------------------------------------
void Vector::operator=(const Vector &vector)
{
  if(size() != vector.size())
    init(vector.size());
  for (int i = 0; i < n; i++)
	 values[i] = vector.values[i];    
}
//-----------------------------------------------------------------------------
void Vector::operator=(real scalar)
{
  for (int i = 0; i < n; i++)
	 values[i] = scalar;    
}
//-----------------------------------------------------------------------------
void Vector::operator+=(real scalar) 
{
  for (int i = 0;i < n; i++)
	 values[i] += scalar;
}
//-----------------------------------------------------------------------------
void Vector::operator+=(const Vector &vector)
{
  for (int i=0;i<n;i++)
	 values[i] += vector.values[i];
}
//-----------------------------------------------------------------------------
void Vector::operator-=(const Vector &vector)
{
  for (int i=0;i<n;i++)
	 values[i] -= vector.values[i];
}
//-----------------------------------------------------------------------------
void Vector::operator*=(real scalar)
{
  for (int i=0;i<n;i++)
	 values[i] *= scalar; 
}
//-----------------------------------------------------------------------------
real Vector::operator*(const Vector &vector)
{
  real sum = 0.0;
  for (int i=0;i<n;i++)
	 sum += values[i] * vector.values[i];

  return sum;
}
//-----------------------------------------------------------------------------
real Vector::norm()
{
  return norm(2);
}
//-----------------------------------------------------------------------------
real Vector::norm(int i)
{
  real norm = 0.0; 

  switch(i){
  case 0:
    // max-norm
    for (int i=0; i<n; i++)
		if ( fabs(values[i]) > norm )
		  norm = fabs(values[i]);
    return norm;
    break;
  case 1:
    // l1-norm
    for (int i=0; i<n; i++)
		norm += fabs(values[i]);
    return norm;
    break;
  case 2:
    // l2-norm
    for (int i=0; i<n; i++)
		norm += values[i] * values[i];
    return sqrt(norm);
    break;
  default:
	 cout << "Unknown vector norm" << endl;
	 exit(1);
  }  

}
//-----------------------------------------------------------------------------
void Vector::add(real scalar, Vector &vector)
{
  for (int i = 0; i < n; i++)
	 values[i] += scalar * vector.values[i];  
}
//-----------------------------------------------------------------------------
void Vector::show() const
{
  cout << "[ ";
  for (int i = 0; i < n; i++)
	 cout << values[i] << " ";
  cout << "]" << endl;
}
//-----------------------------------------------------------------------------
// Additional operators
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream, const Vector& vector)
{
  stream << "[ Vector of size " << vector.size()
			<< ", approximatetly ";
  
  int bytes = vector.bytes();
  
  if ( bytes > 1024*1024 )
	 stream << bytes/1024 << " Mb. ]";
  else if ( bytes > 1024 )
	 stream << bytes/1024 << " kb. ]";
  else
	 stream << bytes << " bytes. ]";
  
  return stream;
}
//-----------------------------------------------------------------------------
