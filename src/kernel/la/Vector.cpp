// (c) 2002 Johan Hoffman & Anders Logg, Chalmers Finite Element Center.
// Licensed under the GNU GPL Version 2.
//
// Contributions by: Georgios Foufas (2002)
//                   Johan Jansson (2003)

#include <dolfin/dolfin_log.h>
#include <dolfin/Vector.h>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
Vector::Vector() : Variable("x", "A vector")
{
  n = 0;
  values = 0;
}
//-----------------------------------------------------------------------------
Vector::Vector(int size) : Variable("x", "A vector")
{
  values = 0;
  
  init(size);
}
//-----------------------------------------------------------------------------
Vector::Vector(const Vector& vector) : Variable(vector.name(), vector.label())
{
  n = vector.n;
  values = new real[n];
  for (int i = 0; i < n; i++)
    values[i] = vector.values[i];
}
//-----------------------------------------------------------------------------
Vector::Vector(real x0) : Variable("x", "A vector")
{
  values = 0;
  
  init(1);
  values[0] = x0;
}
//-----------------------------------------------------------------------------
Vector::Vector(real x0, real x1) : Variable("x", "A vector")
{
  values = 0;
  
  init(2);
  values[0] = x0;
  values[1] = x1;
}
//-----------------------------------------------------------------------------
Vector::Vector(real x0, real x1, real x2) : Variable("x", "A vector")
{
  values = 0;
  
  init(3);
  values[0] = x0;
  values[1] = x1;
  values[2] = x2;
  
  std::cout << "Initialising vector of length 3" << std::endl;
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
int Vector::bytes() const
{
  return sizeof(Vector) + n*sizeof(real);
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
