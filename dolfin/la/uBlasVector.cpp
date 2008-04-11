// Copyright (C) 2006-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006-2008.
// Modified by Kent-Andre Mardal 2008.
//
// First added:  2006-04-04
// Last changed: 2008-04-10

#include <sstream>
#include <iomanip>

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
uBlasVector::uBlasVector()
  : GenericVector(),
    Variable("x", "a dense vector"),
    x(0)
{
  //Do nothing
}
//-----------------------------------------------------------------------------
uBlasVector::uBlasVector(uint N)
  : GenericVector(),
    Variable("x", "a dense vector"),
    x(N)
{
  // Clear vector
  x.clear();
}
//-----------------------------------------------------------------------------
uBlasVector::~uBlasVector()
{
  //Do nothing
}
//-----------------------------------------------------------------------------
void uBlasVector::init(uint N)
{
  if( x.size() == N)
  {
    x.clear();
    return;
  }
 
  x.resize(N, false);
  x.clear();
}
//-----------------------------------------------------------------------------
uBlasVector* uBlasVector::create() const
{
  return new uBlasVector();
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
void uBlasVector::div(const uBlasVector& y)
{
  x = ublas::element_div(x, y.vec());
}
//-----------------------------------------------------------------------------
void uBlasVector::axpy(real a, const GenericVector& y_)
{
  const uBlasVector* y = dynamic_cast<const uBlasVector*>(y_.instance());  
  if ( !y )  
    error("The vector needs to be of type uBlasVector"); 
  if ( size() != y->size() )  
    error("Vectors must be of same size.");

  x += a*y->vec(); 
}
//-----------------------------------------------------------------------------
void uBlasVector::mult(const real a)
{
  x *= a;
}
//-----------------------------------------------------------------------------
real uBlasVector::inner(const GenericVector& y_) const
{
  const uBlasVector* y = dynamic_cast<const uBlasVector*>(y_.instance());  
  if (!y)  
    error("The vector needs to be of type uBlasVector"); 
  return ublas::inner_prod(x, y->vec()); 
}
//-----------------------------------------------------------------------------
const uBlasVector& uBlasVector::operator= (real a) 
{ 
  x.ublas_vector::assign(ublas::scalar_vector<double> (x.size(), a));
  return *this;
}
//-----------------------------------------------------------------------------
const uBlasVector& uBlasVector::operator= (const GenericVector& y_) 
{ 
  const uBlasVector* y = dynamic_cast<const uBlasVector*>(y_.instance());  
  if (!y) 
    error("The vector should be of type uBlasVector");  
  
  x = y->vec();
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
const uBlasVector& uBlasVector::operator+= (const GenericVector& y_) 
{ 
  const uBlasVector* y = dynamic_cast<const uBlasVector*>(y_.instance());  
  if (!y)  
    error("The vector needs to be of type uBlasVector"); 

  x += y->vec();
  return *this; 
}
//-----------------------------------------------------------------------------
const uBlasVector& uBlasVector::operator-= (const GenericVector& y_) 
{ 
  const uBlasVector* y = dynamic_cast<const uBlasVector*>(y_.instance());  
  if (!y)  
    error("The vector needs to be of type uBlasVector"); 
  x -= y->vec();
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
#ifdef HAS_PETSC
void uBlasVector::copy(const PETScVector& y, uint off1, uint off2, uint len)
{
  // FIXME: Verify if there's a more efficient implementation
  const real* vals = 0;
  vals = y.array();
  for(uint i = 0; i < len; i++)
    x(i + off1) = vals[i + off2];

  y.restore(vals);
}
#endif
//-----------------------------------------------------------------------------
void uBlasVector::copy(const uBlasVector& y, uint off1, uint off2, uint len)
{
  ublas::subrange(x, off1, off1 + len) = ublas::subrange(y.vec(), off2, off2 + len);
}
//-----------------------------------------------------------------------------
