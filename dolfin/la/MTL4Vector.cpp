// Copyright (C) 2008 Dag Lindbo
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008.
//
// First added:  2008-07-06
// Last changed: 2008-07-20

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
  // Do nothing
}
//-----------------------------------------------------------------------------
void MTL4Vector::init(uint N)
{
  if (this->size() != N) 
    x.change_dim(N);
  x = 0.0;
}
//-----------------------------------------------------------------------------
MTL4Vector* MTL4Vector::copy() const
{
  return new MTL4Vector(*this);
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
void MTL4Vector::apply()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MTL4Vector::disp(uint precision) const
{
  dolfin::cout << "[ ";
  for (uint i = 0; i < size(); ++i)
  {
    std::stringstream entry;
    entry << std::setiosflags(std::ios::scientific);
    entry << std::setprecision(precision);
    entry << x[i] << " ";
    dolfin::cout << entry.str().c_str() << dolfin::endl;
  }
  dolfin::cout << " ]" << endl;
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
  for (uint i = 0; i < size(); i++)
    values[i] = x[i];
}
//-----------------------------------------------------------------------------
void MTL4Vector::set(real* values)
{
  for (uint i = 0; i < size(); i++)
    x[i] = values[i];
}
//-----------------------------------------------------------------------------
void MTL4Vector::add(real* values)
{
  for (uint i = 0; i < size(); i++)
    x(i) += values[i];
}
//-----------------------------------------------------------------------------
void MTL4Vector::get(real* block, uint m, const uint* rows) const
{
  for (uint i = 0; i < m; i++)
    block[i] = x[ rows[i] ];
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
real MTL4Vector::inner(const GenericVector& v) const
{
  // Developers note: The literal template arguments refers to the number 
  // of levels of loop unrolling that is done at compile time.
  return mtl::dot<6>(x, v.down_cast<MTL4Vector>().vec() );
}
//-----------------------------------------------------------------------------
void MTL4Vector::axpy(real a, const GenericVector& v) 
{
  // Developers note: This is a hack. One would like:
  // x += a*v.down_cast<MTL4Vector>().vec(); 
  mtl4_vector vv =  v.down_cast<MTL4Vector>().vec();
  vv *= a;
  x += vv;
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& MTL4Vector::factory() const
{
  return MTL4Factory::instance();
}
//-----------------------------------------------------------------------------
const MTL4Vector& MTL4Vector::operator= (const GenericVector& v)
{
  x = v.down_cast<MTL4Vector>().vec();
  return *this; 
}
//-----------------------------------------------------------------------------
const MTL4Vector& MTL4Vector::operator= (real a)
{
  x = a;
  return *this; 
}
//-----------------------------------------------------------------------------
const MTL4Vector& MTL4Vector::operator/= (real a)
{
  x /= a;
  return *this; 
}
//-----------------------------------------------------------------------------
const MTL4Vector& MTL4Vector::operator*= (real a)
{
  x *= a;
  return *this;
}
//-----------------------------------------------------------------------------
const MTL4Vector& MTL4Vector::operator= (const MTL4Vector& v)
{
  x = v.vec();
  return *this; 
}
//-----------------------------------------------------------------------------
const MTL4Vector& MTL4Vector::operator+= (const GenericVector& v)
{
  x += v.down_cast<MTL4Vector>().vec();
  return *this;
}
//-----------------------------------------------------------------------------
const MTL4Vector& MTL4Vector::operator-= (const GenericVector& v)
{
  x -= v.down_cast<MTL4Vector>().vec();
  return *this;
}
//-----------------------------------------------------------------------------
real MTL4Vector::norm(VectorNormType type) const
{
  switch (type) 
  {
    case l1:
      return mtl::one_norm(x);
    case l2:
      return mtl::two_norm(x);
    case linf:
      return mtl::infinity_norm(x);
    default:
      error("Requested vector norm type for uBlasVector unknown");
  }
  return 0.0;
}
//-----------------------------------------------------------------------------
real MTL4Vector::min() const
{
  return mtl::min(x);
}
//-----------------------------------------------------------------------------
real MTL4Vector::max() const
{
  return mtl::max(x);
}
//-----------------------------------------------------------------------------

#endif
