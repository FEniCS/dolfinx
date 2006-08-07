// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2006-04-04
// Last changed: 2006-08-07

#include <dolfin/dolfin_log.h>
#include <boost/numeric/ublas/vector.hpp>
#include <dolfin/uBlasVector.h>

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
void uBlasVector::set(const real block[], const int pos[], int n)
{
  for(int i = 0; i < n; ++i)
    (*this)(pos[i]) = block[i];
}
//-----------------------------------------------------------------------------
void uBlasVector::add(const real block[], const int pos[], int n)
{
  for(int i = 0; i < n; ++i)
    (*this)(pos[i]) += block[i];
}
//-----------------------------------------------------------------------------
void uBlasVector::get(real block[], const int pos[], int n) const
{
  for(int i = 0; i < n; ++i)
    block[i] = (*this)(pos[i]);
}
//-----------------------------------------------------------------------------
real uBlasVector::norm(NormType type) const
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
real uBlasVector::sum() const
{
  return sum();
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
const uBlasVector& uBlasVector::operator= (real a) 
{ 
  this->assign(ublas::scalar_vector<double> (this->size(), a));
  return *this;
}
//-----------------------------------------------------------------------------
void uBlasVector::disp(uint precision) const
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
