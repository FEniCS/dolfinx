// Copyright (C) 2006-2008 Garth N. Wells
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
// Modified by Anders Logg 2006-2010.
// Modified by Kent-Andre Mardal 2008.
// Modified by Martin Sandve Alnes 2008.
//
// First added:  2006-04-04
// Last changed: 2011-01-14

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/unordered_set.hpp>

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Array.h>
#include "uBLASVector.h"
#include "uBLASFactory.h"
#include "LinearAlgebraFactory.h"

#ifdef HAS_PETSC
#include "PETScVector.h"
#endif

using namespace dolfin;

//-----------------------------------------------------------------------------
uBLASVector::uBLASVector(): x(new ublas_vector(0))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBLASVector::uBLASVector(uint N): x(new ublas_vector(N))
{
  // Set all entries to zero
  x->clear();
}
//-----------------------------------------------------------------------------
uBLASVector::uBLASVector(const uBLASVector& x): x(new ublas_vector(*(x.x)))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBLASVector::uBLASVector(boost::shared_ptr<ublas_vector> x) : x(x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBLASVector::~uBLASVector()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBLASVector* uBLASVector::copy() const
{
  return new uBLASVector(*this);
}
//-----------------------------------------------------------------------------
void uBLASVector::resize(uint N)
{
  if (x->size() == N)
    return;
  x->resize(N, false);

  // Set vector to zero to prevent random numbers entering the vector.
  // Fixes this bug: https://bugs.launchpad.net/dolfin/+bug/594954
  x->clear();
}
//-----------------------------------------------------------------------------
void uBLASVector::resize(std::pair<uint, uint> range)
{
  if (range.first != 0)
  {
    dolfin_error("uBLASVector.cpp",
                 "resize uBLAS vector",
                 "Distributed vectors not supported by uBLAS backend");
  }

  resize(range.second - range.first);
}
//-----------------------------------------------------------------------------
void uBLASVector::resize(std::pair<uint, uint> range,
                    const std::vector<uint>& ghost_indices)
{
  if (range.first != 0)
  {
    dolfin_error("uBLASVector.cpp",
                 "resize uBLAS vector",
                 "Distributed vectors not supported by uBLAS backend");
  }

  if (ghost_indices.size() != 0)
  {
    dolfin_error("uBLASVector.cpp",
                 "resize uBLAS vector",
                 "Distributed vectors not supported by uBLAS backend");
  }

  resize(range.second - range.first);
}
//-----------------------------------------------------------------------------
dolfin::uint uBLASVector::size() const
{
  return x->size();
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> uBLASVector::local_range() const
{
  return std::make_pair(0, size());
}
//-----------------------------------------------------------------------------
bool uBLASVector::owns_index(uint i) const
{
  if (i < size())
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
void uBLASVector::get_local(double* block, uint m, const uint* rows) const
{
  for (uint i = 0; i < m; i++)
    block[i] = (*x)(rows[i]);
}
//-----------------------------------------------------------------------------
void uBLASVector::get_local(Array<double>& values) const
{
  values.resize(size());
  for (uint i = 0; i < size(); i++)
    values[i] = (*x)(i);
}
//-----------------------------------------------------------------------------
void uBLASVector::set_local(const Array<double>& values)
{
  dolfin_assert(values.size() == size());
  for (uint i = 0; i < size(); i++)
    (*x)(i) = values[i];
}
//-----------------------------------------------------------------------------
void uBLASVector::add_local(const Array<double>& values)
{
  dolfin_assert(values.size() == size());
  for (uint i = 0; i < size(); i++)
    (*x)(i) += values[i];
}
//-----------------------------------------------------------------------------
void uBLASVector::gather(GenericVector& x, const Array<uint>& indices) const
{
  not_working_in_parallel("uBLASVector::gather)");

  const uint _size = indices.size();
  dolfin_assert(this->size() >= _size);

  x.resize(_size);
  ublas_vector& _x = x.down_cast<uBLASVector>().vec();
  for (uint i = 0; i < _size; i++)
    _x(i) = (*this->x)(indices[i]);
}
//-----------------------------------------------------------------------------
void uBLASVector::gather(Array<double>& x, const Array<uint>& indices) const
{
  not_working_in_parallel("uBLASVector::gather)");

  const uint _size = indices.size();
  x.resize(_size);
  dolfin_assert(x.size() == _size);
  for (uint i = 0; i < _size; i++)
    x[i] = (*this->x)(indices[i]);
}
//-----------------------------------------------------------------------------
void uBLASVector::gather_on_zero(Array<double>& x) const
{
  not_working_in_parallel("uBLASVector::gather_on_zero)");

  get_local(x);
}
//-----------------------------------------------------------------------------
void uBLASVector::set(const double* block, uint m, const uint* rows)
{
  for (uint i = 0; i < m; i++)
    (*x)(rows[i]) = block[i];
}
//-----------------------------------------------------------------------------
void uBLASVector::add(const double* block, uint m, const uint* rows)
{
  for (uint i = 0; i < m; i++)
    (*x)(rows[i]) += block[i];
}
//-----------------------------------------------------------------------------
void uBLASVector::apply(std::string mode)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void uBLASVector::zero()
{
  x->clear();
}
//-----------------------------------------------------------------------------
double uBLASVector::norm(std::string norm_type) const
{
  if (norm_type == "l1")
    return norm_1(*x);
  else if (norm_type == "l2")
    return norm_2(*x);
  else if (norm_type == "linf")
    return norm_inf(*x);
  else
  {
    dolfin_error("uBLASVector.cpp",
                 "compute norm of uBLAS vector",
                 "Unknown norm type (\"%s\")", norm_type.c_str());
  }

  return 0.0;
}
//-----------------------------------------------------------------------------
double uBLASVector::min() const
{
  double value = *std::min_element(x->begin(), x->end());
  return value;
}
//-----------------------------------------------------------------------------
double uBLASVector::max() const
{
  double value = *std::max_element(x->begin(), x->end());
  return value;
}
//-----------------------------------------------------------------------------
double uBLASVector::sum() const
{
  return ublas::sum(*x);
}
//-----------------------------------------------------------------------------
double uBLASVector::sum(const Array<uint>& rows) const
{
  boost::unordered_set<uint> row_set;
  double _sum = 0.0;
  for (uint i = 0; i < rows.size(); ++i)
  {
    const uint index = rows[i];
    dolfin_assert(index < size());
    if (row_set.find(index) == row_set.end())
    {
      _sum += (*x)[index];
      row_set.insert(index);
    }
  }
  return _sum;
}
//-----------------------------------------------------------------------------
void uBLASVector::axpy(double a, const GenericVector& y)
{
  if (size() != y.size())
  {
    dolfin_error("uBLASVector.cpp",
                 "perform axpy operation with uBLAS vector",
                 "Vectors are not of the same size");
  }

  (*x) += a * y.down_cast<uBLASVector>().vec();
}
//-----------------------------------------------------------------------------
void uBLASVector::abs()
{
  dolfin_assert(x);
  const uint size = x->size();
  for (uint i = 0; i < size; i++)
    (*x)[i] = std::abs((*x)[i]);
}
//-----------------------------------------------------------------------------
double uBLASVector::inner(const GenericVector& y) const
{
  return ublas::inner_prod(*x, y.down_cast<uBLASVector>().vec());
}
//-----------------------------------------------------------------------------
const GenericVector& uBLASVector::operator= (const GenericVector& v)
{
  *this = v.down_cast<uBLASVector>();
  return *this;
}
//-----------------------------------------------------------------------------
const uBLASVector& uBLASVector::operator= (const uBLASVector& v)
{
  if (size() != v.size())
  {
    dolfin_error("uBLASVector.cpp",
                 "assign one vector to another",
                 "Vectors must be of the same length when assigning. "
                 "Consider using the copy constructor instead");
  }

  assert(x);
  *x = v.vec();
  return *this;
}
//-----------------------------------------------------------------------------
const uBLASVector& uBLASVector::operator= (double a)
{
  x->ublas_vector::assign(ublas::scalar_vector<double> (x->size(), a));
  return *this;
}
//-----------------------------------------------------------------------------
const uBLASVector& uBLASVector::operator*= (const double a)
{
  (*x) *= a;
  return *this;
}
//-----------------------------------------------------------------------------
const uBLASVector& uBLASVector::operator*= (const GenericVector& y)
{
  *x = ublas::element_prod(*x,y.down_cast<uBLASVector>().vec());
  return *this;
}
//-----------------------------------------------------------------------------
const uBLASVector& uBLASVector::operator/= (const double a)
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
std::string uBLASVector::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    s << "[";
    for (ublas_vector::const_iterator it = x->begin(); it != x->end(); ++it)
    {
      std::stringstream entry;
      entry << std::setiosflags(std::ios::scientific);
      entry << std::setprecision(16);
      entry << *it << " ";
      s << entry.str() << std::endl;
    }
    s << "]";
  }
  else
  {
    s << "<uBLASVector of size " << size() << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& uBLASVector::factory() const
{
  return uBLASFactory<>::instance();
}
//-----------------------------------------------------------------------------
