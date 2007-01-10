// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2006-04-04
// Last changed: 2006-12-12

#include <dolfin/dolfin_log.h>
#include <boost/numeric/ublas/vector.hpp>
#include <dolfin/uBlasVector.h>

#ifdef HAVE_PETSC_H
#include <dolfin/PETScVector.h>
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
uBlasVector::uBlasVector(const uint N)
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
void uBlasVector::init(const uint N)
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
void uBlasVector::set(const real block[], const int pos[], const int n)
{
  for(int i = 0; i < n; ++i)
    (*this)(pos[i]) = block[i];
}
//-----------------------------------------------------------------------------
void uBlasVector::add(const real block[], const int pos[], const int n)
{
  for(int i = 0; i < n; ++i)
    (*this)(pos[i]) += block[i];
}
//-----------------------------------------------------------------------------
void uBlasVector::get(real block[], const int pos[], const int n) const
{
  for(int i = 0; i < n; ++i)
    block[i] = (*this)(pos[i]);
}
//-----------------------------------------------------------------------------
real uBlasVector::norm(const NormType type) const
{
  switch (type) {
  case l1:
    return norm_1(*this);
  case l2:
    return norm_2(*this);
  case linf:
    return norm_inf(*this);
  default:
    dolfin_error("Requested vector norm type for uBlasVector unknown");
  }
  return norm_inf(*this);
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
void uBlasVector::axpy(const real a, const uBlasVector& x)
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
const uBlasVector& uBlasVector::operator= (const real a) 
{ 
  this->assign(ublas::scalar_vector<double> (this->size(), a));
  return *this;
}
//-----------------------------------------------------------------------------
void uBlasVector::disp(const uint precision) const
{
  dolfin::cout << "[ ";
  for (uint i = 0; i < size(); i++)
    cout << (*this)(i) << " ";
  dolfin::cout << "]" << endl;
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
#ifdef HAVE_PETSC_H
void uBlasVector::copy(const PETScVector& y, const uint off1, const uint off2, 
                       const uint len)
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
void uBlasVector::copy(const uBlasVector& y, const uint off1, const uint off2, 
                       const uint len)
{
  uBlasVector& x = *this;

  subrange(x, off1, off1 + len) = subrange(y, off2, off2 + len);
}
//-----------------------------------------------------------------------------
