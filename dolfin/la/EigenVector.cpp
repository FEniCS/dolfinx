// Copyright (C) 2015 Garth N. Wells
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

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_set>

#include <dolfin/log/log.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/Array.h>
#include "EigenVector.h"
#include "EigenFactory.h"
#include "GenericLinearAlgebraFactory.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
EigenVector::EigenVector() : EigenVector(MPI_COMM_SELF)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EigenVector::EigenVector(MPI_Comm comm) : _x(new Eigen::VectorXd),
                                          _mpi_comm(comm)
{
  // Check size of communicator
  check_mpi_size(comm);
}
//-----------------------------------------------------------------------------
EigenVector::EigenVector(MPI_Comm comm, std::size_t N)
  : _x(new Eigen::VectorXd(N)), _mpi_comm(comm)
{
  // Check size of communicator
  check_mpi_size(comm);

  // Zero vector
  _x->setZero();
}
//-----------------------------------------------------------------------------
EigenVector::EigenVector(const EigenVector& x)
  : _x(new Eigen::VectorXd(*(x._x))), _mpi_comm(x._mpi_comm)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EigenVector::EigenVector(std::shared_ptr<Eigen::VectorXd> x) : _x(x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EigenVector::~EigenVector()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericVector> EigenVector::copy() const
{
  std::shared_ptr<GenericVector> y(new EigenVector(*this));
  return y;
}
//-----------------------------------------------------------------------------
bool EigenVector::empty() const
{
  dolfin_assert(_x);
  if (_x->size() == 0)
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
std::size_t EigenVector::size() const
{
  return _x->size();
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t> EigenVector::local_range() const
{
  return std::make_pair(0, size());
}
//-----------------------------------------------------------------------------
bool EigenVector::owns_index(std::size_t i) const
{
  if (i < size())
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
void EigenVector::get_local(double* block, std::size_t m,
                            const dolfin::la_index* rows) const
{
  for (std::size_t i = 0; i < m; i++)
    block[i] = (*_x)(rows[i]);
}
//-----------------------------------------------------------------------------
void EigenVector::get_local(std::vector<double>& values) const
{
  values.resize(size());
  for (std::size_t i = 0; i < size(); i++)
    values[i] = (*_x)(i);
}
//-----------------------------------------------------------------------------
void EigenVector::set_local(const std::vector<double>& values)
{
  dolfin_assert(values.size() == size());
  for (std::size_t i = 0; i < size(); i++)
    (*_x)(i) = values[i];
}
//-----------------------------------------------------------------------------
void EigenVector::add_local(const Array<double>& values)
{
  dolfin_assert(values.size() == size());
  for (std::size_t i = 0; i < size(); i++)
    (*_x)(i) += values[i];
}
//-----------------------------------------------------------------------------
void EigenVector::gather(GenericVector& x,
                         const std::vector<dolfin::la_index>& indices) const
{
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
void EigenVector::gather(std::vector<double>& x,
                         const std::vector<dolfin::la_index>& indices) const
{
  const std::size_t _size = indices.size();
  x.resize(_size);
  dolfin_assert(x.size() == _size);
  for (std::size_t i = 0; i < _size; i++)
    x[i] = (*_x)(indices[i]);
}
//-----------------------------------------------------------------------------
void EigenVector::gather_on_zero(std::vector<double>& x) const
{
  get_local(x);
}
//-----------------------------------------------------------------------------
void EigenVector::set(const double* block, std::size_t m,
                      const dolfin::la_index* rows)
{
  for (std::size_t i = 0; i < m; i++)
    (*_x)(rows[i]) = block[i];
}
//-----------------------------------------------------------------------------
void EigenVector::add(const double* block, std::size_t m,
                      const dolfin::la_index* rows)
{
  for (std::size_t i = 0; i < m; i++)
    (*_x)(rows[i]) += block[i];
}
//-----------------------------------------------------------------------------
void EigenVector::apply(std::string mode)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void EigenVector::zero()
{
  dolfin_assert(_x);
  _x->setZero();
}
//-----------------------------------------------------------------------------
double EigenVector::norm(std::string norm_type) const
{
  dolfin_assert(_x);
  if (norm_type == "l1")
    return _x->lpNorm<1>();
  else if (norm_type == "l2")
    return _x->lpNorm<2>();
  else if (norm_type == "linf")
    return _x->lpNorm<Eigen::Infinity>();
  else
  {
    dolfin_error("EigenVector.cpp",
                 "compute norm of Eigen vector",
                 "Unknown norm type (\"%s\")", norm_type.c_str());
  }

  return 0.0;
}
//-----------------------------------------------------------------------------
double EigenVector::min() const
{
  dolfin_assert(_x);
  return _x->minCoeff();
}
//-----------------------------------------------------------------------------
double EigenVector::max() const
{
  dolfin_assert(_x);
  return _x->maxCoeff();
}
//-----------------------------------------------------------------------------
double EigenVector::sum() const
{
  dolfin_assert(_x);
  return _x->sum();
}
//-----------------------------------------------------------------------------
double EigenVector::sum(const Array<std::size_t>& rows) const
{
  std::unordered_set<std::size_t> row_set;
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
void EigenVector::axpy(double a, const GenericVector& y)
{
  if (size() != y.size())
  {
    dolfin_error("EigenVector.cpp",
                 "perform axpy operation with Eigen vector",
                 "Vectors are not of the same size");
  }

  const Eigen::VectorXd& _y = as_type<const EigenVector>(y).vec();
  (*_x) = _x->array() + a * _y.array();
}
//-----------------------------------------------------------------------------
void EigenVector::abs()
{
  dolfin_assert(_x);
  (*_x) = _x->array().abs();
}
//-----------------------------------------------------------------------------
double EigenVector::inner(const GenericVector& y) const
{
  dolfin_assert(_x);
  return _x->dot(as_type<const EigenVector>(y).vec());
}
//-----------------------------------------------------------------------------
const GenericVector& EigenVector::operator= (const GenericVector& v)
{
  *this = as_type<const EigenVector>(v);
  return *this;
}
//-----------------------------------------------------------------------------
const EigenVector& EigenVector::operator= (const EigenVector& v)
{
  if (size() != v.size())
  {
    dolfin_error("EigenVector.cpp",
                 "assign one vector to another",
                 "Vectors must be of the same length when assigning. "
                 "Consider using the copy constructor instead");
  }

  dolfin_assert(_x);
  *_x = v.vec();
  return *this;
}
//-----------------------------------------------------------------------------
const EigenVector& EigenVector::operator= (double a)
{
  dolfin_assert(_x);
  _x->setConstant(a);
  return *this;
}
//-----------------------------------------------------------------------------
const EigenVector& EigenVector::operator*= (const double a)
{
  dolfin_assert(_x);
  (*_x) *= a;
  return *this;
}
//-----------------------------------------------------------------------------
const EigenVector& EigenVector::operator*= (const GenericVector& y)
{
  dolfin_assert(_x);
  (*_x) = _x->cwiseProduct(as_type<const EigenVector>(y).vec());
  return *this;
}
//-----------------------------------------------------------------------------
const EigenVector& EigenVector::operator/= (const double a)
{
  (*_x) /= a;
  return *this;
}
//-----------------------------------------------------------------------------
const EigenVector& EigenVector::operator+= (const GenericVector& y)
{
  const Eigen::VectorXd& _y = as_type<const EigenVector>(y).vec();
  *_x = _x->array() + _y.array();
  return *this;
}
//-----------------------------------------------------------------------------
const EigenVector& EigenVector::operator+= (double a)
{
  *_x = _x->array() + a;
  return *this;
}
//-----------------------------------------------------------------------------
const EigenVector& EigenVector::operator-= (const GenericVector& y)
{
  const Eigen::VectorXd& _y = as_type<const EigenVector>(y).vec();
  *_x = _x->array() - _y.array();
  return *this;
}
//-----------------------------------------------------------------------------
const EigenVector& EigenVector::operator-= (double a)
{
  *_x = _x->array() - a;
  return *this;
}
//-----------------------------------------------------------------------------
std::string EigenVector::str(bool verbose) const
{
  std::stringstream s;
  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    s << "[";
    for (std::size_t i = 0; i != size(); ++i)
    {
      std::stringstream entry;
      entry << std::setiosflags(std::ios::scientific);
      entry << std::setprecision(16);
      entry << (*_x)[i] << " ";
      s << entry.str() << std::endl;
    }
    s << "]";
  }
  else
    s << "<EigenVector of size " << size() << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
GenericLinearAlgebraFactory& EigenVector::factory() const
{
  return EigenFactory::instance();
}
//-----------------------------------------------------------------------------
void EigenVector::resize(std::size_t N)
{
  if (size() == N)
    return;
  else
    _x->resize(N);

  // Set vector to zero
  _x->setZero();
}
//-----------------------------------------------------------------------------
double* EigenVector::data()
{
  dolfin_assert(_x);
  return _x->data();
}
//-----------------------------------------------------------------------------
const double* EigenVector::data() const
{
  dolfin_assert(_x);
  return _x->data();
}
//-----------------------------------------------------------------------------
