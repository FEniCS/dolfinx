// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-25

#include <stdexcept>
#include <iostream>
#include <cmath>
#include <climits>
#include "Vector.h"
#include "BlockVector.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
BlockVector::BlockVector(uint n_, bool owner_): owner(owner_), n(n_)
{ 
  vectors = new Vector*[n]; 
//  if (owner) 
//  {
    for (uint i=0; i<n; i++) 
      vectors[i] = new Vector(); 
 // }
}
BlockVector::~BlockVector() 
{
  if (owner)
  {
    for (uint i=0; i<n; i++) 
      delete vectors[i]; 
  }
  delete [] vectors; 
}
//-----------------------------------------------------------------------------
BlockVector* BlockVector::copy() const
{
  BlockVector* x= new BlockVector(n); 
  for (uint i=0; i<n; i++)   
    x->set(i,*(this->vec(i).copy())); 
  return x; 
}
//-----------------------------------------------------------------------------
const Vector& BlockVector::vec(uint i) const 
{  
  /*
  if (i < 0 || i >= n) {  
    throw(std::out_of_range("The index is out of range!"));
  }
  std::map<int, Vector*>::iterator iter = vectors.find(i);
  if (iter != vectors.end())  
  {
    return *(iter->second);  
  }
  */
  if (i >= n)  
    error("The index is out of range!");
  return *(vectors[i]); 
}
//-----------------------------------------------------------------------------
Vector& BlockVector::vec(uint i) 
{  
  if (i < 0 || i >= n)
    error("The index is out of range!");
  return *(vectors[i]); 
}
//-----------------------------------------------------------------------------
dolfin::uint BlockVector::size() const 
{  
  return n; 
} 
//-----------------------------------------------------------------------------
void BlockVector::axpy(real a, const BlockVector& x) 
{
  for (uint i=0; i<n; i++) 
    this->vec(i).axpy(a, x.vec(i)); 
}
//-----------------------------------------------------------------------------
real BlockVector::inner(const BlockVector& x) const 
{
  real value = 0.0; 
  for (uint i=0; i<n; i++) 
    value += this->vec(i).inner(x.vec(i)); 
  return value; 
}
//-----------------------------------------------------------------------------
real BlockVector::norm(NormType type) const
{
  real value = 0.0; 
  switch (type) 
  { 
    case l1: 
      for (uint i=0; i<n; i++)  
        value += this->vec(i).norm(type); 
      break; 
    case l2: 
      for (uint i=0; i<n; i++)  
        value += pow(this->vec(i).norm(type), 2); 
      value = sqrt(value); 
      break; 
    default: 
      double tmp= 0.0; 
      for (uint i=0; i<n; i++)  
      {
        tmp = this->vec(i).norm(type); 
        if (tmp > value) 
          value = tmp;    
      }
  }
  return value; 
}
//-----------------------------------------------------------------------------
real BlockVector::min() const
{
  double value = 100000000; //FIXME use MAXFLOAT or something  
  double tmp = 0.0;
  for (uint i=0; i<n; i++)  
  {
    tmp = this->vec(i).min(); 
    if (tmp < value)
      value = tmp; 
  }
  return value;
}
//-----------------------------------------------------------------------------
real BlockVector::max() const
{
  double value = 0.0; //FIXME use MINFLOAT or something  
  double tmp = 0.0;
  for (uint i=0; i<n; i++)  
  {
    tmp = this->vec(i).min(); 
    if (tmp > value)
      value = tmp; 
  }
  return value; 
}
//-----------------------------------------------------------------------------
const BlockVector& BlockVector::operator*= (real a) 
{
  for(uint i=0; i<n; i++) 
    this->vec(i) *= a; 
  return *this; 
}
//-----------------------------------------------------------------------------
const BlockVector& BlockVector::operator/= (real a) 
{
  for(uint i=0; i<n; i++)
    this->vec(i) /= a; 
  return *this; 
}
//-----------------------------------------------------------------------------
const BlockVector& BlockVector::operator+= (const BlockVector& y)
{
  axpy(1.0, y); 
  return *this;  
}
//-----------------------------------------------------------------------------
const BlockVector& BlockVector::operator-= (const BlockVector& y)
{
  axpy(-1.0, y); 
  return *this;  
}
//-----------------------------------------------------------------------------
const BlockVector& BlockVector::operator= (const BlockVector& x)
{
  for(uint i=0; i<n; i++)
    this->vec(i) = x.vec(i); 
  return *this; 
}
//-----------------------------------------------------------------------------
const BlockVector& BlockVector::operator= (real a)
{
  for(uint i=0; i<n; i++)
    this->vec(i) = a; 
  return *this; 
}
//-----------------------------------------------------------------------------
void BlockVector::disp(uint precision) const  
{
  for(uint i=0; i<n; i++) 
  {
    std::cout <<"BlockVector("<<i<<"):"<<std::endl;  
    this->vec(i).disp(precision); 
  }
}
//-----------------------------------------------------------------------------
void BlockVector::set(uint i, Vector& v)
{
//  matrices[i*n+j] = m.copy(); //FIXME. not obvious that copy is the right thing
  vectors[i] = &v; //FIXME. not obvious that copy is the right thing
}
//-----------------------------------------------------------------------------







