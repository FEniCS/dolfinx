// Copyright (C) 2008 Dag Lindbo
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2008-2009.
// Modified by Anders Logg, 2010.
//
// First added:  2008-07-06
// Last changed: 2011-01-14

#ifdef HAS_MTL4

#include <cmath>
#include <boost/unordered_set.hpp>
#include <dolfin/common/Array.h>
#include <dolfin/log/dolfin_log.h>
#include "MTL4Factory.h"
#include "MTL4Vector.h"

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
void MTL4Vector::resize(std::pair<uint, uint> range)
{
  if (range.first != 0)
    error("MTL4Vector does not support distributed vectors.");

  resize(range.second - range.first);
}
//-----------------------------------------------------------------------------
void MTL4Vector::resize(std::pair<uint, uint> range,
                    const std::vector<uint>& ghost_indices)
{
  if (ghost_indices.size() != 0)
    error("MTL4SVector does not support ghost entries.");

  resize(range);
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
bool MTL4Vector::owns_index(uint i) const
{
  if (i < size())
    return true;
  else
    return false;
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
void MTL4Vector::get_local(double* block, uint m, const uint* rows) const
{
  for (uint i = 0; i < m; ++i)
    block[i] = x[ rows[i] ];
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
void MTL4Vector::gather(GenericVector& x, const Array<uint>& indices) const
{
  not_working_in_parallel("MTL4Vector::gather)");

  const uint _size = indices.size();
  assert(this->size() >= _size);

  x.resize(_size);
  mtl4_vector& _x = x.down_cast<MTL4Vector>().vec();
  for (uint i = 0; i < _size; i++)
    _x[i] = this->x[ indices[i] ];
}
//-----------------------------------------------------------------------------
void MTL4Vector::gather(Array<double>& x, const Array<uint>& indices) const
{
  not_working_in_parallel("MTL4Vector::gather)");

  const uint _size = indices.size();
  x.resize(_size);
  assert(x.size() == _size);
  for (uint i = 0; i < _size; i++)
    x[i] = this->x[ indices[i] ];
}
//-----------------------------------------------------------------------------
void MTL4Vector::gather_on_zero(Array<double>& x) const
{
  not_working_in_parallel("MTL4Vector::gather_on_zero)");
  get_local(x);
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
  x += a*v.down_cast<MTL4Vector>().vec();
}
//-----------------------------------------------------------------------------
void MTL4Vector::abs()
{
  for (uint i = 0; i < size(); ++i)
    x[i] = std::abs(x[i]);
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
  if (size() != y.size())
    error("Vectors must be of same size.");

  const mtl4_vector& _y = y.down_cast<MTL4Vector>().vec();
  for (uint i = 0; i < size(); ++i)
    x[i] *= _y[i];
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
  assert(size() > 0);
  return mtl::min(x);
}
//-----------------------------------------------------------------------------
double MTL4Vector::max() const
{
  assert(size() > 0);
  return mtl::max(x);
}
//-----------------------------------------------------------------------------
double MTL4Vector::sum() const
{
  return mtl::sum(x);
}
//-----------------------------------------------------------------------------
double MTL4Vector::sum(const Array<uint>& rows) const
{
  boost::unordered_set<uint> row_set;
  double _sum = 0.0;
  for (uint i = 0; i < rows.size(); ++i)
  {
    const uint index = rows[i];
    assert(index < size());
    if (row_set.find(index) == row_set.end())
    {
      _sum += x[index];
      row_set.insert(index);
    }
  }
  return _sum;
}
//-----------------------------------------------------------------------------

#endif
