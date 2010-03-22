// Copyright (C) 2008 Dag Lindbo
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008-2009.
//
// First added:  2008-07-06
// Last changed: 2009-08-22

#ifdef HAS_MTL4

#include <cmath>
#include <dolfin/common/Array.h>
#include <dolfin/math/dolfin_math.h>
#include <dolfin/log/dolfin_log.h>
#include "MTL4Vector.h"
#include "MTL4Factory.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MTL4Vector::MTL4Vector()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MTL4Vector::MTL4Vector(uint N)
{
  resize(N);
}
//-----------------------------------------------------------------------------
MTL4Vector::MTL4Vector(const MTL4Vector& v)
{
  *this = v;
}
//-----------------------------------------------------------------------------
MTL4Vector::~MTL4Vector()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MTL4Vector::resize(uint N)
{
  if (this->size() != N)
    x.change_dim(N);
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
std::pair<dolfin::uint, dolfin::uint> MTL4Vector::local_range() const
{
  return std::make_pair<uint, uint>(0, size());
}
//-----------------------------------------------------------------------------
void MTL4Vector::zero()
{
  x = 0.0;
}
//-----------------------------------------------------------------------------
void MTL4Vector::apply(std::string mode)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::string MTL4Vector::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    s << "[ ";
    for (uint i = 0; i < size(); ++i)
    {
      std::stringstream entry;
      entry << x[i] << " ";
      s << entry.str().c_str() << std::endl;
    }
    s << " ]" << std::endl;
  }
  else
  {
    s << "<MTL4Vector of size " << size() << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
void MTL4Vector::get_local(Array<double>& values) const
{
  values.resize(size());
  for (uint i = 0; i < size(); i++)
    values[i] = x[i];
}
//-----------------------------------------------------------------------------
void MTL4Vector::set_local(const Array<double>& values)
{
  assert(values.size() == size());
  for (uint i = 0; i < size(); i++)
    x[i] = values[i];
}
//-----------------------------------------------------------------------------
void MTL4Vector::add_local(const Array<double>& values)
{
  assert(values.size() == size());
  for (uint i = 0; i < size(); i++)
    x(i) += values[i];
}
//-----------------------------------------------------------------------------
void MTL4Vector::get(double* block, uint m, const uint* rows) const
{
  for (uint i = 0; i < m; i++)
    block[i] = x[ rows[i] ];
}
//-----------------------------------------------------------------------------
void MTL4Vector::set(const double* block, uint m, const uint* rows)
{
  for (uint i = 0; i < m; i++)
    x[ rows[i] ] = block[i];
}
//-----------------------------------------------------------------------------
void MTL4Vector::add(const double* block, uint m, const uint* rows)
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
double MTL4Vector::inner(const GenericVector& v) const
{
  // Developers note: The literal template arguments refers to the number
  // of levels of loop unrolling that is done at compile time.
  return mtl::dot<6>(x, v.down_cast<MTL4Vector>().vec());
}
//-----------------------------------------------------------------------------
void MTL4Vector::axpy(double a, const GenericVector& v)
{
  // Developers note: This is a hack. One would like:
  // x += a*v.down_cast<MTL4Vector>().vec();
  mtl4_vector vv =  v.down_cast<MTL4Vector>().vec();
  vv *= a;
  x  += vv;
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& MTL4Vector::factory() const
{
  return MTL4Factory::instance();
}
//-----------------------------------------------------------------------------
const GenericVector& MTL4Vector::operator= (const GenericVector& v)
{
  *this = v.down_cast<MTL4Vector>();
  return *this;
}
//-----------------------------------------------------------------------------
const MTL4Vector& MTL4Vector::operator= (double a)
{
  x = a;
  return *this;
}
//-----------------------------------------------------------------------------
const MTL4Vector& MTL4Vector::operator/= (double a)
{
  x /= a;
  return *this;
}
//-----------------------------------------------------------------------------
const MTL4Vector& MTL4Vector::operator*= (double a)
{
  x *= a;
  return *this;
}
//-----------------------------------------------------------------------------
const MTL4Vector& MTL4Vector::operator*= (const GenericVector& y)
{
  if ( size() != y.size() )
    error("Vectors must be of same size.");

  const MTL4Vector& _y = y.down_cast<MTL4Vector>();
  for (uint i = 0; i < size(); ++i)
    x[i] *= _y.vec()[i];
  return *this;
}
//-----------------------------------------------------------------------------
const MTL4Vector& MTL4Vector::operator= (const MTL4Vector& v)
{
  resize(v.size());
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
double MTL4Vector::norm(std::string norm_type) const
{
  if (norm_type == "l1")
    return mtl::one_norm(x);
  else if (norm_type == "l2")
    return mtl::two_norm(x);
  else if (norm_type == "linf")
    return mtl::infinity_norm(x);
  else
    error("Requested vector norm type for MTL4Vector unknown");

  return 0.0;
}
//-----------------------------------------------------------------------------
double MTL4Vector::min() const
{
  return mtl::min(x);
}
//-----------------------------------------------------------------------------
double MTL4Vector::max() const
{
  return mtl::max(x);
}
//-----------------------------------------------------------------------------
double MTL4Vector::sum() const
{
  return mtl::sum(x);
}
//-----------------------------------------------------------------------------
#endif
