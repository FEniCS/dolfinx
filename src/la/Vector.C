#include <Display.hh>
#include "Vector.hh"
#include <math.h>

//-----------------------------------------------------------------------------
Vector::Vector()
{
  n = 0;
  values = 0;
}
//-----------------------------------------------------------------------------
Vector::Vector(int n)
{
  values = 0;

  Resize(n);
}
//-----------------------------------------------------------------------------
Vector::~Vector()
{
  delete [] values;
}
//-----------------------------------------------------------------------------
void Vector::Resize(int n)
{
  if ( values )
	 delete [] values;

  this->n = n;

  values = new real[n];

  for (int i=0;i<n;i++)
	 values[i] = 0.0;
}
//-----------------------------------------------------------------------------
void Vector::CopyTo(Vector* vec)
{
  if (Size()!=vec->Size()) display->InternalError("Vector::operator = ()","Vectors are not the same length");

  for (int i=0; i<n; i++) vec->Set(i,values[i]);    
}
//-----------------------------------------------------------------------------
void Vector::CopyFrom(Vector* vec)
{
  if (Size()!=vec->Size()) display->InternalError("Vector::operator = ()","Vectors are not the same length");

  for (int i=0; i<n; i++) Set(i,vec->Get(i));    
}
//-----------------------------------------------------------------------------
void Vector::SetToConstant(real val)
{
  for (int i=0; i<n; i++) Set(i,val);    
}
//-----------------------------------------------------------------------------
real Vector::operator()(int i)
{
  if ((i<0)||(i>=n))
	 display->InternalError("Vector::operator()","Illegal vector index: %d",i);

  return values[i];
}
//-----------------------------------------------------------------------------
void Vector::Set(int i, real val)
{
  if ((i<0)||(i>=n))
    display->InternalError("Vector::Set()","Illegal vector index: %d",i);

  values[i] = val;
}
//-----------------------------------------------------------------------------
real Vector::Get(int i)
{
  if ((i<0)||(i>=n))
    display->InternalError("Vector::Get()","Illegal vector index: %d",i);
	 
  return values[i];
}
//-----------------------------------------------------------------------------
void Vector::Add(int i, real val)
{
  if ((i<0)||(i>=n))
    display->InternalError("Vector::Add()","Illegal vector index: %d",i);

  values[i] += val;
}
//-----------------------------------------------------------------------------
void Vector::Add(real a, Vector *v)
{
  if ( n != v->Size() )
	 display->InternalError("Vector::Add()","Dimensions don't match: %d != %d.",n,v->Size());

  for (int i=0;i<n;i++)
	 values[i] += a*v->values[i];
}
//-----------------------------------------------------------------------------
void Vector::Mult(int i, real val)
{
  if ((i<0)||(i>=n))
	 display->InternalError("Vector::Add()","Illegal vector index");

  values[i] *= val;
}
//-----------------------------------------------------------------------------
real Vector::Dot(Vector *v)
{
  if ( n != v->Size() )
	 display->InternalError("Vector::Dot()","Dimensions don't match: %d != %d.",n,v->Size());
  
  real sum = 0.0;
  for (int i=0;i<n;i++)
	 sum += values[i]*v->values[i];

  return sum;
}
//-----------------------------------------------------------------------------
real Vector::Norm()
{
  // Returns the l2-norm of the vector
  return Norm(2);
}
//-----------------------------------------------------------------------------
real Vector::Norm(int i)
{
  real norm = 0.0; 

  switch(i){
  case 0:
    // max-norm
    for (int i=0;i<n;i++) if (fabs(values[i]) > norm) norm = fabs(values[i]);
    return norm;
    break;
  case 1:
    // l1-norm
    for (int i=0;i<n;i++) norm += fabs(values[i]);
    return norm;
    break;
  case 2:
    // l2-norm
    for (int i=0;i<n;i++) norm += sqr(values[i]);
    return sqrt(norm);
    break;
  default:
    display->InternalError("Vector::Norm()","This norm is not implemented");
  }  

}
//-----------------------------------------------------------------------------
int Vector::Size()
{
  return n;
}
//-----------------------------------------------------------------------------
void Vector::Display()
{
  printf("Vector of size n = %d: v = [ ",n);
  for (int i=0;i<n;i++)
	 printf("%f ",values[i]);
  printf("]\n");
}
//-----------------------------------------------------------------------------
void Vector::Write(const char *filename)
{
  FILE *fp = fopen(filename,"w");
  if ( !fp )
    display->Error("Unable to write to file \"%s\".",filename);

  for (int i=0;i<(n-1);i++)
    fprintf(fp,"%1.16e ",values[i]);
  fprintf(fp,"%1.16e\n",values[n-1]);

  fclose(fp);
}
//-----------------------------------------------------------------------------
