// (c) 2002 Johan Hoffman & Anders Logg, Chalmers Finite Element Center.
// Licensed under the GNU GPL Version 2.
//
// Modifications by Georgios Foufas (2002)

#include <dolfin/Display.h>
#include <dolfin/Vector.h>
#include <math.h>
#include "utils.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Vector::Vector()
{
  n = 0;
  values = 0;
}
//-----------------------------------------------------------------------------
Vector::Vector(int size)
{
  values = 0;

  resize(size);
}
//-----------------------------------------------------------------------------
int Vector::bytes()
{
  return sizeof(Vector) + n*sizeof(real);
}
//-----------------------------------------------------------------------------
Vector::~Vector()
{
  delete [] values;
}
//-----------------------------------------------------------------------------
void Vector::resize(int size)
{
  if ( values )
	 delete [] values;

  n = size;

  values = new real[n];

  for (int i=0;i<n;i++)
	 values[i] = 0.0;
}
//-----------------------------------------------------------------------------
int Vector::size()
{
  return n;
}
//-----------------------------------------------------------------------------
real& Vector::operator()(int i)
{
  if ( (i<0) || (i>=n) )
	 display->InternalError("Vector::operator ()",
									"Illegal vector index: %d",i);
         
  return values[i];
}
//-----------------------------------------------------------------------------
void Vector::operator=(Vector &vector)
{
  if ( size() != vector.size() )
	 display->InternalError("Vector::operator = ()",
									"Vectors are not the same length");

  for (int i=0; i<n; i++)
	 values[i] = vector.values[i];    
}
//-----------------------------------------------------------------------------
void Vector::operator=(real scalar)
{
  for (int i=0; i<n; i++)
	 values[i] = scalar;    
}
//-----------------------------------------------------------------------------
void Vector::operator+=(real scalar) 
{
  for (int i=0;i<n;i++)
	 values[i] += scalar;
}
//-----------------------------------------------------------------------------
void Vector::operator+=(Vector &vector)
{
  if ( n != vector.size() )
	 display->InternalError("Vector::operator +=",
									"Dimensions don't match: %d != %d.",n,vector.size());

  for (int i=0;i<n;i++)
	 values[i] += vector.values[i];
}
//-----------------------------------------------------------------------------
void Vector::operator*=(real scalar)
{
  for (int i=0;i<n;i++)
	 values[i] *= scalar; 
}
//-----------------------------------------------------------------------------
real Vector::operator*(Vector &vector)
{
  if ( n != vector.size() )
	 display->InternalError("Vector::operator *",
									"Dimensions don't match: %d != %d.",n,vector.size());
  
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
    display->InternalError("Vector::Norm()","This norm is not implemented");
  }  

}
//-----------------------------------------------------------------------------
void Vector::add(real scalar, Vector &vector)
{
  if ( n != vector.size() )
	 display->InternalError("Vector::add",
									"Dimensions don't match: %d != %d.",n,vector.size());
  
  for (int i = 0; i < n; i++)
	 values[i] += scalar * vector.values[i];  
}
//-----------------------------------------------------------------------------
void Vector::show()
{
  cout << "x = [ ";
  for (int i = 0; i < n; i++)
	 cout << values[i] << " ";
  cout << "]" << endl;
}
//-----------------------------------------------------------------------------
namespace dolfin {

  //---------------------------------------------------------------------------
  ostream& operator << (ostream& output, Vector& vector)
  {
	 output << "[ Vector of size " << vector.size()
			  << ", approximatetly ";

	 int bytes = vector.bytes();
	 
	 if ( bytes > 1024*1024 )
		output << bytes/1024 << " Mb.]";
	 else if ( bytes > 1024 )
		output << bytes/1024 << " kb.]";
	 else
		output << bytes << " bytes.]";

	 return output;
  }
  //---------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
