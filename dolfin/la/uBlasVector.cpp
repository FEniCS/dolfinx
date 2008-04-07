// Copyright (C) 2006-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006-2008.
// Modified by Kent-Andre Mardal 2008.
//
// First added:  2006-04-04
// Last changed: 2008-03-19

#include <sstream>
#include <iomanip>

#include <dolfin/log/dolfin_log.h>
#include <boost/numeric/ublas/vector.hpp>
#include "uBlasVector.h"
#include "uBlasFactory.h"

#ifdef HAS_PETSC
#include "PETScVector.h"
#endif

using namespace dolfin;

//-----------------------------------------------------------------------------
uBlasVector::uBlasVector()
  : GenericVector(),
    Variable("x", "a dense vector"),
    ublas_vector()
{
  //Do nothing
}
//-----------------------------------------------------------------------------
uBlasVector::uBlasVector(uint N)
  : GenericVector(),
    Variable("x", "a dense vector"),
    ublas_vector(N)
{
  // Clear matrix (not done by ublas)
  clear();
}
//-----------------------------------------------------------------------------
uBlasVector::~uBlasVector()
{
  //Do nothing
}
//-----------------------------------------------------------------------------
void uBlasVector::init(uint N)
{
  if( this->size() == N)
  {
    clear();
    return;
  }
  
  this->resize(N, false);
  clear();
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
    values[i] = (*this)(i);
}
//-----------------------------------------------------------------------------
void uBlasVector::set(real* values)
{
  for (uint i = 0; i < size(); i++)
    (*this)(i) = values[i];
}
//-----------------------------------------------------------------------------
void uBlasVector::add(real* values)
{
  for (uint i = 0; i < size(); i++)
    (*this)(i) += values[i];
}
//-----------------------------------------------------------------------------
void uBlasVector::get(real* block, uint m, const uint* rows) const
{
  for (uint i = 0; i < m; i++)
    block[i] = (*this)(rows[i]);
}
//-----------------------------------------------------------------------------
void uBlasVector::set(const real* block, uint m, const uint* rows)
{
  for (uint i = 0; i < m; i++)
    (*this)(rows[i]) = block[i];
}
//-----------------------------------------------------------------------------
void uBlasVector::add(const real* block, uint m, const uint* rows)
{
  for (uint i = 0; i < m; i++)
    (*this)(rows[i]) += block[i];
}
//-----------------------------------------------------------------------------
void uBlasVector::apply()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void uBlasVector::zero()
{
  clear();
}
//-----------------------------------------------------------------------------
real uBlasVector::norm(VectorNormType type) const
{
  switch (type) {
  case l1:
    return norm_1(*this);
  case l2:
    return norm_2(*this);
  case linf:
    return norm_inf(*this);
  default:
    error("Requested vector norm type for uBlasVector unknown");
  }
  return norm_inf(*this);
}
void uBlasVector::div(const uBlasVector& y)
{
  uBlasVector& x = *this;
  uint s = size();

  for(uint i = 0; i < s; i++)
  {
    x[i] = x[i] / y[i];
  }
}
//-----------------------------------------------------------------------------
void uBlasVector::add(const GenericVector& x_, real a) {
  const uBlasVector* x = dynamic_cast<const uBlasVector*>(x_.instance());  
  if (!x)  error("The vector needs to be of type uBlasVector"); 
  axpy(a,*x); 
}
//-----------------------------------------------------------------------------
void uBlasVector::axpy(real a, const uBlasVector& x)
{
  uBlasVector& y = *this;
  
  y += a * x;
}
//-----------------------------------------------------------------------------
void uBlasVector::mult(const real a)
{
  uBlasVector& y = *this;
  
  y *= a;
}
//-----------------------------------------------------------------------------
real uBlasVector::inner(const GenericVector& x_) const
{
  const uBlasVector* x = dynamic_cast<const uBlasVector*>(x_.instance());  
  if (!x)  error("The vector needs to be of type uBlasVector"); 
  return inner_prod(*this,*x); 
}
//-----------------------------------------------------------------------------
const uBlasVector& uBlasVector::operator= (real a) 
{ 
  this->assign(ublas::scalar_vector<double> (this->size(), a));
  return *this;
}
//-----------------------------------------------------------------------------
const uBlasVector& uBlasVector::operator= (const GenericVector& x_) 
{ 
  cout << "Assignment in uBlasVector" << endl;

  const uBlasVector* x = dynamic_cast<const uBlasVector*>(x_.instance());  
  if (!x) error("The vector should be of type PETScVector");  
  
  *this = (*x)*1.0; 
  return *this; 
}




//-----------------------------------------------------------------------------
void uBlasVector::disp(uint precision) const
{
  dolfin::cout << "[ ";
  for (ublas_vector::const_iterator it = this->begin(); it != this->end(); ++it)
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

  uBlasVector& x = *this;
  const real* vals = 0;
  vals = y.array();
  for(uint i = 0; i < len; i++)
  {
    x[i + off1] = vals[i + off2];
  }
  y.restore(vals);
}
#endif
//-----------------------------------------------------------------------------
void uBlasVector::copy(const uBlasVector& y, uint off1, uint off2, uint len)
{
  uBlasVector& x = *this;

  subrange(x, off1, off1 + len) = subrange(y, off2, off2 + len);
}
//-----------------------------------------------------------------------------
