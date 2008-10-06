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
  if (owner) 
  {
    for (uint i=0; i<n; i++) 
      vectors[i] = new Vector(); 
  }
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
    x->set(i,*(this->getc(i).copy())); 
  return x; 
}
//-----------------------------------------------------------------------------
SubVector BlockVector::operator()(uint i)
{
  SubVector sv(i,*this);  
  return sv; 
}
//-----------------------------------------------------------------------------
dolfin::uint BlockVector::size() const 
{  
  return n; 
} 
//-----------------------------------------------------------------------------
void BlockVector::axpy(double a, const BlockVector& x) 
{
  for (uint i=0; i<n; i++) 
    this->get(i).axpy(a, x.getc(i)); 
}
//-----------------------------------------------------------------------------
double BlockVector::inner(const BlockVector& x) const 
{
  double value = 0.0; 
  for (uint i=0; i<n; i++) 
    value += this->getc(i).inner(x.getc(i)); 
  return value; 
}
//-----------------------------------------------------------------------------
double BlockVector::norm(NormType type) const
{
  double value = 0.0; 
  switch (type) 
  { 
    case l1: 
      for (uint i=0; i<n; i++)  
        value += this->getc(i).norm(type); 
      break; 
    case l2: 
      for (uint i=0; i<n; i++)  
        value += pow(this->getc(i).norm(type), 2); 
      value = sqrt(value); 
      break; 
    default: 
      double tmp= 0.0; 
      for (uint i=0; i<n; i++)  
      {
        tmp = this->getc(i).norm(type); 
        if (tmp > value) 
          value = tmp;    
      }
  }
  return value; 
}
//-----------------------------------------------------------------------------
double BlockVector::min() const
{
  double value = 100000000; //FIXME use MAXFLOAT or something  
  double tmp = 0.0;
  for (uint i=0; i<n; i++)  
  {
    tmp = this->getc(i).min(); 
    if (tmp < value)
      value = tmp; 
  }
  return value;
}
//-----------------------------------------------------------------------------
double BlockVector::max() const
{
  double value = -1.0; //FIXME use MINFLOAT or something  
  double tmp = 0.0;
  for (uint i=0; i<n; i++)  
  {
    tmp = this->getc(i).min(); 
    if (tmp > value)
      value = tmp; 
  }
  return value; 
}
//-----------------------------------------------------------------------------
const BlockVector& BlockVector::operator*= (double a) 
{
  for(uint i=0; i<n; i++) 
    this->get(i) *= a; 
  return *this; 
}
//-----------------------------------------------------------------------------
const BlockVector& BlockVector::operator/= (double a) 
{
  for(uint i=0; i<n; i++)
    this->get(i) /= a; 
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
    this->get(i) = x.getc(i); 
  return *this; 
}
//-----------------------------------------------------------------------------
const BlockVector& BlockVector::operator= (double a)
{
  for(uint i=0; i<n; i++)
    this->get(i) = a; 
  return *this; 
}
//-----------------------------------------------------------------------------
void BlockVector::disp(uint precision) const  
{
  for(uint i=0; i<n; i++) 
  {
    std::cout <<"BlockVector("<<i<<"):"<<std::endl;  
    this->getc(i).disp(precision); 
  }
}
//-----------------------------------------------------------------------------
void BlockVector::set(uint i, Vector& v)
{
//  matrices[i*n+j] = m.copy(); //FIXME. not obvious that copy is the right thing
  vectors[i] = &v; //FIXME. not obvious that copy is the right thing
}
//-----------------------------------------------------------------------------
const Vector& BlockVector::getc(uint i) const
{
  return *(vectors[i]); 
}
//-----------------------------------------------------------------------------
Vector& BlockVector::get(uint i) 
{
  return *(vectors[i]); 
}
//-----------------------------------------------------------------------------
// SubVector
//-----------------------------------------------------------------------------
SubVector::SubVector(uint n_, BlockVector& bv_) 
  : n(n_),bv(bv_)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SubVector::~SubVector()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const SubVector& SubVector::operator=(Vector& v) 
{
  bv.set(n, v);  
  return *this; 
}
/*
//-----------------------------------------------------------------------------
Vector& SubVector::operator()
{
  return bm.get(row, col); 
}
*/
//-----------------------------------------------------------------------------
