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
// Modified by Anders Logg 2006-2012
// Modified by Kent-Andre Mardal 2008
// Modified by Martin Sandve Alnes 2008
//
// First added:  2006-04-04
// Last changed: 2012-03-15

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/unordered_set.hpp>

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/Array.h>
#include "uBLASVector.h"
#include "uBLASFactory.h"
#include "GenericLinearAlgebraFactory.h"

#ifdef HAS_PETSC
#include "PETScVector.h"
#endif

using namespace dolfin;

//-----------------------------------------------------------------------------
uBLASVector::uBLASVector(std::string type): _x(new ublas_vector(0))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBLASVector::uBLASVector(std::size_t N, std::string type): _x(new ublas_vector(N))
{
  // Set all entries to zero
  _x->clear();
}
//-----------------------------------------------------------------------------
uBLASVector::uBLASVector(const uBLASVector& x): _x(new ublas_vector(*(x._x)))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBLASVector::uBLASVector(boost::shared_ptr<ublas_vector> x) : _x(x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBLASVector::~uBLASVector()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericVector> uBLASVector::copy() const
{
  boost::shared_ptr<GenericVector> y(new uBLASVector(*this));
  return y;
}
//-----------------------------------------------------------------------------
void uBLASVector::resize(std::size_t N)
{
  if (_x->size() == N)
    return;
  _x->resize(N, false);

  // Set vector to zero to prevent random numbers entering the vector.
  // Fixes this bug: https://bugs.launchpad.net/dolfin/+bug/594954
  _x->clear();
}
//-----------------------------------------------------------------------------
void uBLASVector::resize(std::pair<std::size_t, std::size_t> range)
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
void uBLASVector::resize(std::pair<std::size_t, std::size_t> range,
                    const std::vector<std::size_t>& ghost_indices)
{
  if (range.first != 0)
  {
    dolfin_error("uBLASVector.cpp",
                 "resize uBLAS vector",
                 "Distributed vectors not supported by uBLAS backend");
  }

  if (!ghost_indices.empty())
  {
    dolfin_error("uBLASVector.cpp",
                 "resize uBLAS vector",
                 "Distributed vectors not supported by uBLAS backend");
  }

  resize(range.second - range.first);
}
//-----------------------------------------------------------------------------
bool uBLASVector::empty() const
{
  return _x->empty();
}
//-----------------------------------------------------------------------------
std::size_t uBLASVector::size() const
{
  return _x->size();
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t> uBLASVector::local_range() const
{
  return std::make_pair(0, size());
}
//-----------------------------------------------------------------------------
bool uBLASVector::owns_index(std::size_t i) const
{
  if (i < size())
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
void uBLASVector::get_local(double* block, std::size_t m, const dolfin::la_index* rows) const
{
  for (std::size_t i = 0; i < m; i++)
    block[i] = (*_x)(rows[i]);
}
//-----------------------------------------------------------------------------
void uBLASVector::get_local(std::vector<double>& values) const
{
  values.resize(size());
  for (std::size_t i = 0; i < size(); i++)
    values[i] = (*_x)(i);
}
//-----------------------------------------------------------------------------
void uBLASVector::set_local(const std::vector<double>& values)
{
  dolfin_assert(values.size() == size());
  for (std::size_t i = 0; i < size(); i++)
    (*_x)(i) = values[i];
}
//-----------------------------------------------------------------------------
void uBLASVector::add_local(const Array<double>& values)
{
  dolfin_assert(values.size() == size());
  for (std::size_t i = 0; i < size(); i++)
    (*_x)(i) += values[i];
}
//-----------------------------------------------------------------------------
void uBLASVector::gather(GenericVector& x, const std::vector<dolfin::la_index>& indices) const
{
  not_working_in_parallel("uBLASVector::gather)");

  const std::size_t _size = indices.size();
  dolfin_assert(this->size() >= _size);

  x.resize(_size);
  ublas_vector& tmp = as_type<uBLASVector>(x).vec();
  for (std::size_t i = 0; i < _size; i++)
    tmp(i) = (*_x)(indices[i]);
}
//-----------------------------------------------------------------------------
void uBLASVector::gather(std::vector<double>& x,
                         const std::vector<dolfin::la_index>& indices) const
{
  not_working_in_parallel("uBLASVector::gather)");

  const std::size_t _size = indices.size();
  x.resize(_size);
  dolfin_assert(x.size() == _size);
  for (std::size_t i = 0; i < _size; i++)
    x[i] = (*_x)(indices[i]);
}
//-----------------------------------------------------------------------------
void uBLASVector::gather_on_zero(std::vector<double>& x) const
{
  not_working_in_parallel("uBLASVector::gather_on_zero)");

  get_local(x);
}
//-----------------------------------------------------------------------------
void uBLASVector::set(const double* block, std::size_t m, const dolfin::la_index* rows)
{
  for (std::size_t i = 0; i < m; i++)
    (*_x)(rows[i]) = block[i];
}
//-----------------------------------------------------------------------------
void uBLASVector::add(const double* block, std::size_t m, const dolfin::la_index* rows)
{
  for (std::size_t i = 0; i < m; i++)
    (*_x)(rows[i]) += block[i];
}
//-----------------------------------------------------------------------------
void uBLASVector::apply(std::string mode)
{
  Timer("Apply (vector)");

  // Do nothing
}
//-----------------------------------------------------------------------------
void uBLASVector::zero()
{
  _x->clear();
}
//-----------------------------------------------------------------------------
double uBLASVector::norm(std::string norm_type) const
{
  if (norm_type == "l1")
    return norm_1(*_x);
  else if (norm_type == "l2")
    return norm_2(*_x);
  else if (norm_type == "linf")
    return norm_inf(*_x);
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
  double value = *std::min_element(_x->begin(), _x->end());
  return value;
}
//-----------------------------------------------------------------------------
double uBLASVector::max() const
{
  double value = *std::max_element(_x->begin(), _x->end());
  return value;
}
//-----------------------------------------------------------------------------
double uBLASVector::sum() const
{
  return ublas::sum(*_x);
}
//-----------------------------------------------------------------------------
double uBLASVector::sum(const Array<std::size_t>& rows) const
{
  boost::unordered_set<std::size_t> row_set;
  double _sum = 0.0;
  for (std::size_t i = 0; i < rows.size(); ++i)
  {
    const std::size_t index = rows[i];
    dolfin_assert(index < size());
    if (row_set.find(index) == row_set.end())
    {
      _sum += (*_x)[index];
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

  (*_x) += a * as_type<const uBLASVector>(y).vec();
}
//-----------------------------------------------------------------------------
void uBLASVector::abs()
{
  dolfin_assert(_x);
  const std::size_t size = _x->size();
  for (std::size_t i = 0; i < size; i++)
    (*_x)[i] = std::abs((*_x)[i]);
}
//-----------------------------------------------------------------------------
double uBLASVector::inner(const GenericVector& y) const
{
  return ublas::inner_prod(*_x, as_type<const uBLASVector>(y).vec());
}
//-----------------------------------------------------------------------------
const GenericVector& uBLASVector::operator= (const GenericVector& v)
{
  *this = as_type<const uBLASVector>(v);
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

  assert(_x);
  *_x = v.vec();
  return *this;
}
//-----------------------------------------------------------------------------
const uBLASVector& uBLASVector::operator= (double a)
{
  _x->ublas_vector::assign(ublas::scalar_vector<double> (_x->size(), a));
  return *this;
}
//-----------------------------------------------------------------------------
const uBLASVector& uBLASVector::operator*= (const double a)
{
  (*_x) *= a;
  return *this;
}
//-----------------------------------------------------------------------------
const uBLASVector& uBLASVector::operator*= (const GenericVector& y)
{
  *_x = ublas::element_prod(*_x, as_type<const uBLASVector>(y).vec());
  return *this;
}
//-----------------------------------------------------------------------------
const uBLASVector& uBLASVector::operator/= (const double a)
{
  (*_x) /= a;
  return *this;
}
//-----------------------------------------------------------------------------
const uBLASVector& uBLASVector::operator+= (const GenericVector& y)
{
  *_x += as_type<const uBLASVector>(y).vec();
  return *this;
}
//-----------------------------------------------------------------------------
const uBLASVector& uBLASVector::operator+= (double a)
{
  boost::numeric::ublas::scalar_vector<double> _a(_x->size(), a);
  *_x += _a;
  return *this;
}
//-----------------------------------------------------------------------------
const uBLASVector& uBLASVector::operator-= (const GenericVector& y)
{
  *_x -= as_type<const uBLASVector>(y).vec();
  return *this;
}
//-----------------------------------------------------------------------------
const uBLASVector& uBLASVector::operator-= (double a)
{
  boost::numeric::ublas::scalar_vector<double> _a(_x->size(), a);
  *_x -= _a;
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
    for (ublas_vector::const_iterator it = _x->begin(); it != _x->end(); ++it)
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
    s << "<uBLASVector of size " << size() << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
GenericLinearAlgebraFactory& uBLASVector::factory() const
{
  return uBLASFactory<>::instance();
}
//-----------------------------------------------------------------------------
