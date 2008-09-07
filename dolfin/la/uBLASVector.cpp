// Copyright (C) 2006-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006-2008.
// Modified by Kent-Andre Mardal 2008.
// Modified by Martin Sandve Alnes 2008.
//
// First added:  2006-04-04
// Last changed: 2008-09-07

#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include <dolfin/log/dolfin_log.h>
#include <boost/numeric/ublas/vector.hpp>
#include "uBLASVector.h"
#include "uBLASFactory.h"
#include "LinearAlgebraFactory.h"

#ifdef HAS_PETSC
#include "PETScVector.h"
#endif

using namespace dolfin;

//-----------------------------------------------------------------------------
uBLASVector::uBLASVector(): Variable("x", "uBLAS vector"), x(new ublas_vector(0))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBLASVector::uBLASVector(uint N): Variable("x", "uBLAS vector"), 
                                  x(new ublas_vector(N))
{
  // Clear vector
  x->clear();
}
//-----------------------------------------------------------------------------
uBLASVector::uBLASVector(const uBLASVector& x): Variable("x", "uBLAS vector"), 
                         x(new ublas_vector(*(x.x)))
{
  //Do nothing
}
//-----------------------------------------------------------------------------
uBLASVector::~uBLASVector()
{
  //Do nothing
}
//-----------------------------------------------------------------------------
void uBLASVector::init(uint N)
{
  if(x->size() == N)
  {
    x->clear();
    return;
  }
 
  x->resize(N, false);
  x->clear();
}
//-----------------------------------------------------------------------------
dolfin::uint uBLASVector::size() const
{
  return x->size();
}
//-----------------------------------------------------------------------------
uBLASVector* uBLASVector::copy() const
{
  return new uBLASVector(*this);
}
//-----------------------------------------------------------------------------
void uBLASVector::get(real* values) const
{
  for (uint i = 0; i < size(); i++)
    values[i] = (*x)(i);
}
//-----------------------------------------------------------------------------
void uBLASVector::set(real* values)
{
  for (uint i = 0; i < size(); i++)
    (*x)(i) = values[i];
}
//-----------------------------------------------------------------------------
void uBLASVector::add(real* values)
{
  for (uint i = 0; i < size(); i++)
    (*x)(i) += values[i];
}
//-----------------------------------------------------------------------------
void uBLASVector::get(real* block, uint m, const uint* rows) const
{
  for (uint i = 0; i < m; i++)
    block[i] = (*x)(rows[i]);
}
//-----------------------------------------------------------------------------
void uBLASVector::set(const real* block, uint m, const uint* rows)
{
  for (uint i = 0; i < m; i++)
    (*x)(rows[i]) = block[i];
}
//-----------------------------------------------------------------------------
void uBLASVector::add(const real* block, uint m, const uint* rows)
{
  for (uint i = 0; i < m; i++)
    (*x)(rows[i]) += block[i];
}
//-----------------------------------------------------------------------------
void uBLASVector::apply()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void uBLASVector::zero()
{
  x->clear();
}
//-----------------------------------------------------------------------------
real uBLASVector::norm(NormType type) const
{
  switch (type) 
  {
    case l1:
      return norm_1(*x);
    case l2:
      return norm_2(*x);
    case linf:
      return norm_inf(*x);
    default:
      error("Requested vector norm type for uBLASVector unknown");
  }
  return 0.0;
}
//-----------------------------------------------------------------------------
real uBLASVector::min() const
{
  real value = *std::min_element(x->begin(), x->end());
  return value;
}
//-----------------------------------------------------------------------------
real uBLASVector::max() const
{
  real value = *std::max_element(x->begin(), x->end());
  return value;
}
//-----------------------------------------------------------------------------
void uBLASVector::axpy(real a, const GenericVector& y)
{
  if ( size() != y.size() )  
    error("Vectors must be of same size.");

  (*x) += a * y.down_cast<uBLASVector>().vec();
}
//-----------------------------------------------------------------------------
real uBLASVector::inner(const GenericVector& y) const
{
  return ublas::inner_prod(*x, y.down_cast<uBLASVector>().vec());
}
//-----------------------------------------------------------------------------
const GenericVector& uBLASVector::operator= (const GenericVector& y) 
{ 
  *x = y.down_cast<uBLASVector>().vec();
  return *this; 
}
//-----------------------------------------------------------------------------
const uBLASVector& uBLASVector::operator= (const uBLASVector& y) 
{ 
  *x = y.vec();
  return *this; 
}
//-----------------------------------------------------------------------------
const uBLASVector& uBLASVector::operator= (real a) 
{ 
  x->ublas_vector::assign(ublas::scalar_vector<double> (x->size(), a));
  return *this; 
}
//-----------------------------------------------------------------------------
const uBLASVector& uBLASVector::operator*= (const real a) 
{ 
  (*x) *= a;
  return *this;     
}
//-----------------------------------------------------------------------------
const uBLASVector& uBLASVector::operator/= (const real a) 
{ 
  (*x) /= a;
  return *this;     
}
//-----------------------------------------------------------------------------
const uBLASVector& uBLASVector::operator+= (const GenericVector& y) 
{ 
  *x += y.down_cast<uBLASVector>().vec();
  return *this; 
}
//-----------------------------------------------------------------------------
const uBLASVector& uBLASVector::operator-= (const GenericVector& y) 
{ 
  *x -= y.down_cast<uBLASVector>().vec();
  return *this; 
}
//-----------------------------------------------------------------------------
void uBLASVector::disp(uint precision) const
{
  dolfin::cout << "[ ";
  for (ublas_vector::const_iterator it = x->begin(); it != x->end(); ++it)
  {
    std::stringstream entry;
    entry << std::setiosflags(std::ios::scientific);
    entry << std::setprecision(precision);
    entry << *it << " ";
    dolfin::cout << entry.str().c_str() << dolfin::endl;
  }
  dolfin::cout << " ]" << endl;
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const uBLASVector& x)
{
  // Check if vector has been defined
  if ( x.size() == 0 )
  {
    stream << "[ uBLASVector (empty) ]";
    return stream;
  }
  stream << "[ uBLASVector of size " << x.size() << " ]";

  return stream;
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& uBLASVector::factory() const
{
  return uBLASFactory<>::instance();
}
//-----------------------------------------------------------------------------
