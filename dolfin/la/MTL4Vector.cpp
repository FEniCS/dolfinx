// Copyright (C) 2008 Dag Lindbo
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-07-06
// Last changed:  2008-07-06

#ifdef HAS_MTL4

#include <cstring>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <cmath>
#include <dolfin/math/dolfin_math.h>
#include <dolfin/log/dolfin_log.h>
#include "MTL4Vector.h"
#include "MTL4Factory.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MTL4Vector::MTL4Vector():
  Variable("x", "a sparse vector")
{
}
//-----------------------------------------------------------------------------
MTL4Vector::MTL4Vector(uint N):
  Variable("x", "a sparse vector") 
{
  init(N);
}
//-----------------------------------------------------------------------------
MTL4Vector::MTL4Vector(const MTL4Vector& v):
  Variable("x", "a vector")
{
  *this = v;
}
//-----------------------------------------------------------------------------
MTL4Vector::~MTL4Vector()
{
}
//-----------------------------------------------------------------------------
void MTL4Vector::init(uint N)
{
  if (this->size() != N) 
    x.change_dim(N);
}
//-----------------------------------------------------------------------------
MTL4Vector* MTL4Vector::copy() const
{
  error("MTL4::copy not implemented yet");
  return (MTL4Vector*) NULL;
}
//-----------------------------------------------------------------------------
dolfin::uint MTL4Vector::size() const
{
  return mtl::num_rows(x);
}
//-----------------------------------------------------------------------------
void MTL4Vector::zero()
{
  x = 0.0;
}
//-----------------------------------------------------------------------------
void MTL4Vector::apply(FinalizeType finaltype)
{
}
//-----------------------------------------------------------------------------
void MTL4Vector::disp(uint precision) const
{
  error("MTL4::disp not implemented yet");
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const MTL4Vector& x)
{
  error("MTL4::operator<< not implemented yet");
  return stream;
}
//-----------------------------------------------------------------------------
void MTL4Vector::get(real* values) const
{
  error("MTL4::get not implemented yet");
}
//-----------------------------------------------------------------------------
void MTL4Vector::set(real* values)
{
  error("MTL4::set not implemented yet");
}
//-----------------------------------------------------------------------------
void MTL4Vector::add(real* values)
{
  error("MTL4::add not implemented yet");
}
//-----------------------------------------------------------------------------
void MTL4Vector::get(real* block, uint m, const uint* rows) const
{
  error("MTL4Vector::get not implemented yet");
}
//-----------------------------------------------------------------------------
void MTL4Vector::set(const real* block, uint m, const uint* rows)
{
  for (uint i = 0; i < m; i++)
    x[ rows[i] ] = block[i];
}
//-----------------------------------------------------------------------------
void MTL4Vector::add(const real* block, uint m, const uint* rows)
{
  for (uint i = 0; i < m; i++)
    x[ rows[i] ] += block[i];
}
//-----------------------------------------------------------------------------
const mtl4_vector& MTL4Vector::vec() const
{
  return x;
}
//-----------------------------------------------------------------------------
mtl4_vector& MTL4Vector::vec()
{
  return x;
}
//-----------------------------------------------------------------------------
real MTL4Vector::inner(const GenericVector& y) const
{
  error("MTL4::inner not implemented yet");
  return 0.0;
}
//-----------------------------------------------------------------------------
void MTL4Vector::axpy(real a, const GenericVector& y) 
{
  error("MTL4::axpy not implemented yet");
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& MTL4Vector::factory() const
{
  return MTL4Factory::instance();
}
//-----------------------------------------------------------------------------
const MTL4Vector& MTL4Vector::operator= (const GenericVector& v)
{
  error("MTL4::operator=(vec) not implemented yet");
  return *this; 
}
//-----------------------------------------------------------------------------
const MTL4Vector& MTL4Vector::operator= (real a)
{
  error("MTL4::operator=(real) not implemented yet");
  return *this; 
}
//-----------------------------------------------------------------------------
const MTL4Vector& MTL4Vector::operator/= (real a)
{
  error("MTL4::operator/=(real) not implemented yet");
  return *this; 
}
//-----------------------------------------------------------------------------
const MTL4Vector& MTL4Vector::operator*= (real a)
{
  error("MTL4::operator*= not implemented yet");
  return *this;
}
//-----------------------------------------------------------------------------
const MTL4Vector& MTL4Vector::operator= (const MTL4Vector& v)
{
  error("MTL4::operator=(vec) not implemented yet");
  return *this; 
}

//-----------------------------------------------------------------------------
const MTL4Vector& MTL4Vector::operator+= (const GenericVector& y)
{
  error("MTL4::operator+= not implemented yet");
  return *this;
}
//-----------------------------------------------------------------------------
const MTL4Vector& MTL4Vector::operator-= (const GenericVector& y)
{
  error("MTL4::operator-= not implemented yet");
  return *this;
}
//-----------------------------------------------------------------------------
real MTL4Vector::norm(VectorNormType type) const
{
  error("MTL4::norm not implemented yet");
  return 0.0;
}
//-----------------------------------------------------------------------------
real MTL4Vector::min() const
{
  error("MTL4::min not implemented yet");
  return 0.0;
}
//-----------------------------------------------------------------------------
real MTL4Vector::max() const
{
  error("MTL4::max not implemented yet");
  return 0.0;
}
//-----------------------------------------------------------------------------

#endif
