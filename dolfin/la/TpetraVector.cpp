// Copyright (C) 2014 Chris Richardson
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
// First added: 2014

#ifdef HAS_TRILINOS

#include <cmath>
#include <numeric>
#include <dolfin/common/Timer.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Set.h>
#include <dolfin/log/dolfin_log.h>
#include "TpetraVector.h"
#include "TpetraFactory.h"
#include <dolfin/common/MPI.h>


using namespace dolfin;

//-----------------------------------------------------------------------------
TpetraVector::TpetraVector() : _x(NULL)
{
}
//-----------------------------------------------------------------------------
TpetraVector::TpetraVector(MPI_Comm comm, std::size_t N)
  : _x(NULL)
{
  _init(comm, N);
}
//-----------------------------------------------------------------------------
TpetraVector::TpetraVector(const TpetraVector& v) : _x(NULL)
{

}
//-----------------------------------------------------------------------------
TpetraVector::~TpetraVector()
{
}
//-----------------------------------------------------------------------------
void TpetraVector::zero()
{
  dolfin_assert(!_x.is_null());
  _x->putScalar(0.0);
  this->apply("insert");
}
//-----------------------------------------------------------------------------
void TpetraVector::apply(std::string mode)
{
  dolfin_assert(!_x.is_null());
  // Do apply
}
//-----------------------------------------------------------------------------
MPI_Comm TpetraVector::mpi_comm() const
{
  // FIXME: can this be cleaned up somehow?
  Teuchos::RCP<const Teuchos::Comm<int> > _comm 
    = _x->getMap()->getComm();

  const Teuchos::MpiComm<int>& _mpi_comm 
    = static_cast<const Teuchos::MpiComm<int>& >(*_comm);

  return *(_mpi_comm.getRawMpiComm());
}
//-----------------------------------------------------------------------------
std::string TpetraVector::str(bool verbose) const
{
  if (_x.is_null())
    return "<Uninitialized TpetraVector>";

  std::stringstream s;
  if (verbose)
  {
    s << "< " <<_x->description() << " >";
  }
  else
    s << "<TpetraVector of size " << size() << ">";
  
  return s.str();
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericVector> TpetraVector::copy() const
{
  return std::shared_ptr<GenericVector>(new TpetraVector(*this));
}
//-----------------------------------------------------------------------------
void TpetraVector::init(MPI_Comm comm, std::size_t N)
{
  _init(comm, N);
}
//-----------------------------------------------------------------------------
void TpetraVector::init(MPI_Comm comm,
                        std::pair<std::size_t, std::size_t> range)
{
  int N = 0;
  _init(comm, N);
}
//-----------------------------------------------------------------------------
void TpetraVector::init(MPI_Comm comm,
                        std::pair<std::size_t, std::size_t> range,
                        const std::vector<std::size_t>& local_to_global_map,
                        const std::vector<la_index>& ghost_indices)
{
  int N = 0;
  _init(comm, N);
}
//-----------------------------------------------------------------------------
bool TpetraVector::empty() const
{
  return (size() == 0);
}
//-----------------------------------------------------------------------------
std::size_t TpetraVector::size() const
{
  return _x->getGlobalLength();
}
//-----------------------------------------------------------------------------
std::size_t TpetraVector::local_size() const
{
  return _x->getLocalLength();
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t> TpetraVector::local_range() const
{
  dolfin_assert(!_x.is_null());
  Teuchos::RCP<const map_type> xmap(_x->getMap());  
  return std::make_pair(xmap->getMinGlobalIndex(),
                        xmap->getMaxGlobalIndex());
}
//-----------------------------------------------------------------------------
bool TpetraVector::owns_index(std::size_t i) const
{
  dolfin_assert(!_x.is_null());
  Teuchos::RCP<const map_type> xmap(_x->getMap());  
  return xmap->isNodeLocalElement(i);
}
//-----------------------------------------------------------------------------
void TpetraVector::get(double* block, std::size_t m,
                       const dolfin::la_index* rows) const
{
  
}
//-----------------------------------------------------------------------------
void TpetraVector::get_local(double* block, std::size_t m,
                             const dolfin::la_index* rows) const
{
}
//-----------------------------------------------------------------------------
void TpetraVector::set(const double* block, std::size_t m,
                       const dolfin::la_index* rows)
{
  dolfin_assert(!_x.is_null());
  for (std::size_t i = 0; i != m; ++i)
    _x->replaceGlobalValue(rows[i], block[i]);
}
//-----------------------------------------------------------------------------
void TpetraVector::set_local(const double* block, std::size_t m,
                             const dolfin::la_index* rows)
{
  dolfin_assert(!_x.is_null());
  for (std::size_t i = 0; i != m; ++i)
    _x->replaceLocalValue(rows[i], block[i]);
}
//-----------------------------------------------------------------------------
void TpetraVector::add(const double* block, std::size_t m,
                       const dolfin::la_index* rows)
{
  dolfin_assert(!_x.is_null());
  for (std::size_t i = 0; i != m; ++i)
    _x->sumIntoGlobalValue(rows[i], block[i]);
}
//-----------------------------------------------------------------------------
void TpetraVector::add_local(const double* block, std::size_t m,
                             const dolfin::la_index* rows)
{
  dolfin_assert(!_x.is_null());
  for (std::size_t i = 0; i != m; ++i)
    _x->sumIntoLocalValue(rows[i], block[i]);
}
//-----------------------------------------------------------------------------
void TpetraVector::get_local(std::vector<double>& values) const
{
}
//-----------------------------------------------------------------------------
void TpetraVector::set_local(const std::vector<double>& values)
{
  dolfin_assert(!_x.is_null());
  const std::size_t local_size = local_range().second - local_range().first;
  if (values.size() != local_size)
  {
    dolfin_error("TpetraVector.cpp",
                 "set local values of Tpetra vector",
                 "Size of values array is not equal to local vector size");
  }

  if (local_size == 0)
    return;

  // Build array of local indices
  std::vector<dolfin::la_index> rows(local_size);
  for (std::size_t i = 0; i < local_size; ++i)
    rows[i] = i;

  set_local(values.data(), local_size, rows.data());
}
//-----------------------------------------------------------------------------
void TpetraVector::add_local(const Array<double>& values)
{
  dolfin_assert(!_x.is_null());
  const std::size_t local_size = local_range().second - local_range().first;
  if (values.size() != local_size)
  {
    dolfin_error("TpetraVector.cpp",
                 "add local values to Tpetra vector",
                 "Size of values array is not equal to local vector size");
  }

  if (local_size == 0)
    return;

  // Build array of local indices
  std::vector<dolfin::la_index> rows(local_size);
  for (std::size_t i = 0; i < local_size; ++i)
    rows[i] = i;

  add_local(values.data(), local_size, rows.data());
}
//-----------------------------------------------------------------------------
void TpetraVector::gather(GenericVector& x,
                          const std::vector<dolfin::la_index>& indices) const
{
}
//-----------------------------------------------------------------------------
void TpetraVector::gather(std::vector<double>& x,
                          const std::vector<dolfin::la_index>& indices) const
{
}
//-----------------------------------------------------------------------------
void TpetraVector::gather_on_zero(std::vector<double>& x) const 
{
}
//-----------------------------------------------------------------------------
void TpetraVector::axpy(double a, const GenericVector& y)
{
  dolfin_assert(!_x.is_null());
  const TpetraVector& _y = as_type<const TpetraVector>(y);
  dolfin_assert(!_y._x.is_null());
  _x->update(1.0, *_y._x, a);
}
//-----------------------------------------------------------------------------
void TpetraVector::abs()
{
  // FIXME: check this is OK
  _x->abs(*_x);
}
//-----------------------------------------------------------------------------
double TpetraVector::inner(const GenericVector& y) const
{
  dolfin_assert(!_x.is_null());
  const TpetraVector& _y = as_type<const TpetraVector>(y);
  dolfin_assert(!_y._x.is_null());
  return _x->dot(*_y._x);
}
//-----------------------------------------------------------------------------
double TpetraVector::norm(std::string norm_type) const
{
  dolfin_assert(!_x.is_null());
  typedef Tpetra::Vector<>::mag_type mag_type;
  const mag_type _norm = _x->norm2();
  return _norm;
}
//-----------------------------------------------------------------------------
double TpetraVector::min() const
{
  dolfin_assert(!_x.is_null());
  return 0.0;
}
//-----------------------------------------------------------------------------
double TpetraVector::max() const
{
  dolfin_assert(!_x.is_null());
  return 0.0;
}
//-----------------------------------------------------------------------------
double TpetraVector::sum() const
{
  dolfin_assert(!_x.is_null());
  return 0.0;
}
//-----------------------------------------------------------------------------
double TpetraVector::sum(const Array<std::size_t>& rows) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
const TpetraVector& TpetraVector::operator*= (double a)
{
  dolfin_assert(!_x.is_null());
  _x->scale(a);
  return *this;
}
//-----------------------------------------------------------------------------
const TpetraVector& TpetraVector::operator*= (const GenericVector& y) 
{
  dolfin_assert(!_x.is_null());
  //  const TpetraVector& _y = as_type<const TpetraVector>(y);
  //  _x->scale(*_y._x);
  return *this;
}
//-----------------------------------------------------------------------------
const TpetraVector& TpetraVector::operator/= (double a) 
{
  dolfin_assert(!_x.is_null());
  dolfin_assert(a != 0.0);
  const double b = 1.0/a;
  (*this) *= b;
  return *this;
}
//-----------------------------------------------------------------------------
const TpetraVector& TpetraVector::operator+= (const GenericVector& x) 
{
  dolfin_assert(!_x.is_null());
  return *this;
}
//-----------------------------------------------------------------------------
const TpetraVector& TpetraVector::operator+= (double a) 
{
  dolfin_assert(!_x.is_null());
  return *this;
}
//-----------------------------------------------------------------------------
const TpetraVector& TpetraVector::operator-= (const GenericVector& x) 
{
  dolfin_assert(!_x.is_null());
  axpy(-1.0, x);
  return *this;
}
//-----------------------------------------------------------------------------
const TpetraVector& TpetraVector::operator-= (double a) 
{
  dolfin_assert(!_x.is_null());
  (*this) += -a;
  return *this;
}
//-----------------------------------------------------------------------------
const GenericVector& TpetraVector::operator= (const GenericVector& v)
{
  *this = as_type<const TpetraVector>(v);
  return *this;
}
//-----------------------------------------------------------------------------
const TpetraVector& TpetraVector::operator= (double a) 
{
  dolfin_assert(!_x.is_null());
  _x->putScalar(a);
  return *this;
}
//-----------------------------------------------------------------------------
const TpetraVector& TpetraVector::operator= (const TpetraVector& x)
{
  return *this;
}
//-----------------------------------------------------------------------------
GenericLinearAlgebraFactory& TpetraVector::factory() const
{
  return TpetraFactory::instance();
}
//-----------------------------------------------------------------------------
void TpetraVector::_init(MPI_Comm comm,
                         std::size_t N)
{
  if (!_x.is_null())
    error("TpetraVector cannot be initialized more than once.");

  // Make a Trilinos version of the MPI Comm
  Teuchos::RCP<const Teuchos::Comm<int> > 
    _comm(new Teuchos::MpiComm<int>(comm));

  // Mapping across processes
  Teuchos::RCP<map_type> _map = Teuchos::rcp(new map_type(N, 0, _comm));

  // Vector
  _x = Teuchos::rcp(new vector_type(_map));
  
}
//-----------------------------------------------------------------------------
Teuchos::RCP<vector_type> TpetraVector::vec() const
{
  return _x;
}
//-----------------------------------------------------------------------------
#endif
