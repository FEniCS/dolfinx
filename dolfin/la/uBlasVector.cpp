// Copyright (C) 2006-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006-2008.
// Modified by Kent-Andre Mardal 2008.
// Modified by Martin Sandve Alnes 2008.
//
// First added:  2006-04-04
// Last changed: 2008-04-28

#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include <dolfin/log/dolfin_log.h>
#include <boost/numeric/ublas/vector.hpp>
#include "uBlasVector.h"
#include "uBlasFactory.h"
#include "LinearAlgebraFactory.h"

#ifdef HAS_PETSC
#include "PETScVector.h"
#endif

using namespace dolfin;

//-----------------------------------------------------------------------------
uBlasVector::uBlasVector():
    Variable("x", "uBLAS vector"), x(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasVector::uBlasVector(uint N):
    Variable("x", "uBLAS vector"), x(N)
{
  // Clear vector
  x.clear();
}
//-----------------------------------------------------------------------------
uBlasVector::uBlasVector(const uBlasVector& x):
  Variable("x", "uBLAS vector"), x(x.x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasVector::~uBlasVector()
{
  //Do nothing
}
//-----------------------------------------------------------------------------
void uBlasVector::init(uint N)
{
  if(x.size() == N)
  {
    x.clear();
    return;
  }
 
  x.resize(N, false);
  x.clear();
}
//-----------------------------------------------------------------------------
uBlasVector* uBlasVector::copy() const
{
  return new uBlasVector(*this);
}
//-----------------------------------------------------------------------------
void uBlasVector::get(real* values) const
{
  for (uint i = 0; i < size(); i++)
    values[i] = x(i);
}
//-----------------------------------------------------------------------------
void uBlasVector::set(real* values)
{
  for (uint i = 0; i < size(); i++)
    x(i) = values[i];
}
//-----------------------------------------------------------------------------
void uBlasVector::add(real* values)
{
  for (uint i = 0; i < size(); i++)
    x(i) += values[i];
}
//-----------------------------------------------------------------------------
void uBlasVector::get(real* block, uint m, const uint* rows) const
{
  for (uint i = 0; i < m; i++)
    block[i] = x(rows[i]);
}
//-----------------------------------------------------------------------------
void uBlasVector::set(const real* block, uint m, const uint* rows)
{
  for (uint i = 0; i < m; i++)
    x(rows[i]) = block[i];
}
//-----------------------------------------------------------------------------
void uBlasVector::add(const real* block, uint m, const uint* rows)
{
  for (uint i = 0; i < m; i++)
    x(rows[i]) += block[i];
}
//-----------------------------------------------------------------------------
void uBlasVector::apply()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void uBlasVector::zero()
{
  x.clear();
}
//-----------------------------------------------------------------------------
real uBlasVector::norm(VectorNormType type) const
{
  switch (type) {
  case l1:
    return norm_1(x);
  case l2:
    return norm_2(x);
  case linf:
    return norm_inf(x);
  default:
    error("Requested vector norm type for uBlasVector unknown");
  }
  return norm_inf(x);
}
//-----------------------------------------------------------------------------
real uBlasVector::min() const
{
  real value = *std::min_element(x.begin(), x.end());
  return value;
}
//-----------------------------------------------------------------------------
real uBlasVector::max() const
{
  real value = *std::max_element(x.begin(), x.end());
  return value;
}
//-----------------------------------------------------------------------------
void uBlasVector::axpy(real a, const GenericVector& y)
{
  if ( size() != y.size() )  
    error("Vectors must be of same size.");

  x += a * y.down_cast<uBlasVector>().vec();
}
//-----------------------------------------------------------------------------
real uBlasVector::inner(const GenericVector& y) const
{
  return ublas::inner_prod(x, y.down_cast<uBlasVector>().vec());
}
//-----------------------------------------------------------------------------
const GenericVector& uBlasVector::operator= (const GenericVector& y) 
{ 
  x = y.down_cast<uBlasVector>().vec();
  return *this; 
}
//-----------------------------------------------------------------------------
const uBlasVector& uBlasVector::operator= (const uBlasVector& y) 
{ 
  x = y.vec();
  return *this; 
}
//-----------------------------------------------------------------------------
const uBlasVector& uBlasVector::operator*= (const real a) 
{ 
  x *= a;
  return *this;     
}
//-----------------------------------------------------------------------------
const uBlasVector& uBlasVector::operator/= (const real a) 
{ 
  x /= a;
  return *this;     
}
//-----------------------------------------------------------------------------
const uBlasVector& uBlasVector::operator+= (const GenericVector& y) 
{ 
  x += y.down_cast<uBlasVector>().vec();
  return *this; 
}
//-----------------------------------------------------------------------------
const uBlasVector& uBlasVector::operator-= (const GenericVector& y) 
{ 
  x -= y.down_cast<uBlasVector>().vec();
  return *this; 
}
//-----------------------------------------------------------------------------
void uBlasVector::disp(uint precision) const
{
  dolfin::cout << "[ ";
  for (ublas_vector::const_iterator it = x.begin(); it != x.end(); ++it)
  {
    std::stringstream entry;
    entry << std::setiosflags(std::ios::scientific);
    entry << std::setprecision(precision);
    entry << *it << " ";
    dolfin::cout << entry.str().c_str() << dolfin::endl;
  }
  dolfin::cout << " ]";
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const uBlasVector& x)
{
  // Check if vector has been defined
  if ( x.size() == 0 )
  {
    stream << "[ uBlasVector (empty) ]";
    return stream;
  }
  stream << "[ uBlasVector of size " << x.size() << " ]";

  return stream;
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& uBlasVector::factory() const
{
  return uBlasFactory::instance();
}
//-----------------------------------------------------------------------------
