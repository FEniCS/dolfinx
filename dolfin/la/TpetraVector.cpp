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
// First added: Nov 2014

#ifdef HAS_TRILINOS

#include<cstdio>

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
  init(comm, N);
}
//-----------------------------------------------------------------------------
TpetraVector::TpetraVector(const TpetraVector& v) : _x(NULL)
{
  if (v._x.is_null())
    return;

  // Create with same map
  Teuchos::RCP<const map_type> vmap(v._x->getMap());
  _x = Teuchos::rcp(new vector_type(vmap));

  _x->assign(*v._x);
}
//-----------------------------------------------------------------------------
TpetraVector::~TpetraVector()
{
}
//-----------------------------------------------------------------------------
void TpetraVector::zero()
{
  dolfin_assert(!_x.is_null());
  std::cout << "zero()\n";

  _x->putScalar(0.0);
}
//-----------------------------------------------------------------------------
void TpetraVector::apply(std::string mode)
{
  dolfin_assert(!_x.is_null());
  std::cout << "Apply called with: " << mode << "\n";

  std::cout << "Is one to one? " << _x->getMap()->isOneToOne() << "\n";

  if(_x->getMap()->isOneToOne())
    return;

  // Make a one-to-one map from xmap, and a vector based on it
  Teuchos::RCP<const map_type> xmap(_x->getMap());
  Teuchos::RCP<const map_type>
    ymap = Tpetra::createOneToOne(xmap);
  Teuchos::RCP<vector_type> y(new vector_type(ymap));

  // Export from overlapping map x, to non-overlapping map y
  Tpetra::Export<global_ordinal_type> exporter(xmap, ymap);

  // Forward export to reduction vector y
  if (mode == "add")
  {
    y->doExport(*_x, exporter, Tpetra::ADD);
  }
  else if (mode == "insert")
  {
    y->doExport(*_x, exporter, Tpetra::INSERT);
  }

  // Reverse import to put values back into _x
  _x->doImport(*y, exporter, Tpetra::INSERT);

}
//-----------------------------------------------------------------------------
MPI_Comm TpetraVector::mpi_comm() const
{
  // Unwrap MPI_Comm
  const Teuchos::RCP<const Teuchos::MpiComm<int> > _mpi_comm
    = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int> >
    (_x->getMap()->getComm());

  return *(_mpi_comm->getRawMpiComm());
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
  std::pair<std::size_t, std::size_t> range = MPI::local_range(comm, N);
    std::vector<dolfin::la_index> local_to_global_map;
    _init(comm, range, local_to_global_map);
}
//-----------------------------------------------------------------------------
void TpetraVector::init(MPI_Comm comm,
                        std::pair<std::size_t, std::size_t> range)
{
  std::vector<dolfin::la_index> local_to_global_map;
  _init(comm, range, local_to_global_map);
}
//-----------------------------------------------------------------------------
void TpetraVector::init(MPI_Comm comm,
                        std::pair<std::size_t, std::size_t> range,
                        const std::vector<std::size_t>& local_to_global_map,
                        const std::vector<la_index>& ghost_indices)
{
  std::vector<dolfin::la_index> _global_map(local_to_global_map.begin(),
                                            local_to_global_map.end());
  _init(comm, range, _global_map);
}
//-----------------------------------------------------------------------------
bool TpetraVector::empty() const
{
  return (size() == 0);
}
//-----------------------------------------------------------------------------
std::size_t TpetraVector::size() const
{
  if (_x.is_null())
    return 0;
  return (_x->getMap()->getMaxAllGlobalIndex() + 1);
}
//-----------------------------------------------------------------------------
std::size_t TpetraVector::local_size() const
{
  if (_x.is_null())
    return 0;
  return _x->getLocalLength();
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t> TpetraVector::local_range() const
{
  dolfin_assert(!_x.is_null());
  Teuchos::RCP<const map_type> xmap(_x->getMap());

  // FIXME: wrong with ghost entries
  return std::make_pair(xmap->getMinGlobalIndex(),
                        xmap->getMaxGlobalIndex());
}
//-----------------------------------------------------------------------------
bool TpetraVector::owns_index(std::size_t i) const
{
  dolfin_assert(!_x.is_null());

  Teuchos::RCP<const map_type> xmap(_x->getMap());
  int mpi_rank = xmap->getComm()->getRank();

  // FIXME: inefficient? call to getRemoteIndexList requires communication
  std::vector<int> node_list(local_size());
  Teuchos::ArrayView<int> _node_list(node_list);
  xmap->getRemoteIndexList(xmap->getNodeElementList(), _node_list);

  // First check if global index exists on this process
  // Second check if this process is the owner
  const local_ordinal_type idx = xmap->getLocalElement(i);

  bool status;

  if (idx == Teuchos::OrdinalTraits<local_ordinal_type>::invalid())
    status = false;
  else
    status = (node_list[idx] == mpi_rank);

  return status;
}
//-----------------------------------------------------------------------------
void TpetraVector::get(double* block, std::size_t m,
                       const dolfin::la_index* rows) const
{
  dolfin_assert(!_x.is_null());

  Teuchos::RCP<const map_type> xmap(_x->getMap());
  Teuchos::ArrayRCP<const scalar_type> xarr = _x->getData();

  for (std::size_t i = 0; i != m; ++i)
  {
    const local_ordinal_type idx = xmap->getLocalElement(rows[i]);
    if (idx != Teuchos::OrdinalTraits<local_ordinal_type>::invalid())
      block[i] = xarr[idx];
  }
}
//-----------------------------------------------------------------------------
void TpetraVector::get_local(double* block, std::size_t m,
                             const dolfin::la_index* rows) const
{
  dolfin_assert(!_x.is_null());
  Teuchos::ArrayRCP<const scalar_type> arr = _x->getData();
  for (std::size_t i = 0; i!=m; ++i)
    block[i] = arr[rows[i]];
}
//-----------------------------------------------------------------------------
void TpetraVector::set(const double* block, std::size_t m,
                       const dolfin::la_index* rows)
{
  dolfin_assert(!_x.is_null());
  for (std::size_t i = 0; i != m; ++i)
  {
    if(_x->getMap()->isNodeGlobalElement(rows[i]))
      _x->replaceGlobalValue(rows[i], block[i]);
  }
}
//-----------------------------------------------------------------------------
void TpetraVector::set_local(const double* block, std::size_t m,
                             const dolfin::la_index* rows)
{
  dolfin_assert(!_x.is_null());
  for (std::size_t i = 0; i != m; ++i)
  {
    if(_x->getMap()->isNodeLocalElement(rows[i]))
      _x->replaceLocalValue(rows[i], block[i]);
  }
}
//-----------------------------------------------------------------------------
void TpetraVector::add(const double* block, std::size_t m,
                       const dolfin::la_index* rows)
{
  dolfin_assert(!_x.is_null());
  for (std::size_t i = 0; i != m; ++i)
  {
    if(_x->getMap()->isNodeGlobalElement(rows[i]))
      _x->sumIntoGlobalValue(rows[i], block[i]);
  }
}
//-----------------------------------------------------------------------------
void TpetraVector::add_local(const double* block, std::size_t m,
                             const dolfin::la_index* rows)
{
  dolfin_assert(!_x.is_null());

  for (std::size_t i = 0; i != m; ++i)
  {
    if(_x->getMap()->isNodeLocalElement(rows[i]))
      _x->sumIntoLocalValue(rows[i], block[i]);
  }
}
//-----------------------------------------------------------------------------
void TpetraVector::get_local(std::vector<double>& values) const
{
  dolfin_assert(!_x.is_null());
  values.resize(local_size());
  Teuchos::ArrayRCP<const scalar_type> arr = _x->getData();
  std::copy(arr.get(), arr.get() + values.size(), values.begin());
}
//-----------------------------------------------------------------------------
void TpetraVector::set_local(const std::vector<double>& values)
{
  dolfin_assert(!_x.is_null());
  const std::size_t num_values = local_size();
  if (values.size() != num_values)
  {
    dolfin_error("TpetraVector.cpp",
                 "set local values of Tpetra vector",
                 "Size of values array is not equal to local vector size");
  }

  if (num_values == 0)
    return;

  Teuchos::ArrayRCP<scalar_type> arr = _x->getDataNonConst();
  std::copy(values.begin(), values.end(), arr.get());
}
//-----------------------------------------------------------------------------
void TpetraVector::add_local(const Array<double>& values)
{
  dolfin_assert(!_x.is_null());

  const std::size_t num_values = local_size();
  if (values.size() != num_values)
  {
    dolfin_error("TpetraVector.cpp",
                 "add local values to Tpetra vector",
                 "Size of values array is not equal to local vector size");
  }

  for (std::size_t i = 0; i != num_values; ++i)
    _x->sumIntoLocalValue(i, values[i]);
}
//-----------------------------------------------------------------------------
void TpetraVector::gather(GenericVector& y,
                          const std::vector<dolfin::la_index>& indices) const
{
  dolfin_assert(!_x.is_null());

  // FIXME: not working?

  TpetraVector& _y = as_type<TpetraVector>(y);

  const std::pair<std::size_t, std::size_t> range(0, indices.size());

  if (_y._x.is_null())
    _y._init(MPI_COMM_SELF, range, indices);
  else if (y.size() != indices.size() || MPI::size(y.mpi_comm()) != 0)
  {
    dolfin_error("TpetraVector.cpp",
                 "gather vector entries",
                 "Cannot re-initialize gather vector. Must be empty, or have correct size and be a local vector");
  }

  const Tpetra::Export<global_ordinal_type>
    exporter(_x->getMap(), _y._x->getMap());
  _y._x->doExport(*_x, exporter, Tpetra::INSERT);
}
//-----------------------------------------------------------------------------
void TpetraVector::gather(std::vector<double>& x,
                          const std::vector<dolfin::la_index>& indices) const
{
  x.resize(indices.size());
  TpetraVector y;
  gather(y, indices);
  dolfin_assert(y.local_size() == x.size());
  y.get_local(x);
}
//-----------------------------------------------------------------------------
void TpetraVector::gather_on_zero(std::vector<double>& v) const
{
  dolfin_assert(!_x.is_null());

  if (_x->getMap()->getComm()->getRank() == 0)
    v.resize(size());
  else
    v.resize(0);

  // Create map with elements only on process zero
  Teuchos::RCP<map_type> ymap(new map_type(size(),
                                           v.size(), 0,
                                           _x->getMap()->getComm()));
  Teuchos::RCP<vector_type> y(new vector_type(ymap));

  // Export from overlapping vector x to non-overlapping vector y
  const Tpetra::Export<global_ordinal_type>
    exporter(_x->getMap(), y->getMap());
  y->doExport(*_x, exporter, Tpetra::INSERT);
  Teuchos::ArrayRCP<const scalar_type> yarr(y->getData());
  std::copy(yarr.get(), yarr.get() + v.size(), v.begin());
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
  dolfin_assert(!_x.is_null());
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
  Teuchos::ArrayRCP<const scalar_type> arr = _x->getData();
  double min_local
    = *std::min_element(arr.get(), arr.get() + arr.size());

  return MPI::min(mpi_comm(), min_local);
}
//-----------------------------------------------------------------------------
double TpetraVector::max() const
{
  dolfin_assert(!_x.is_null());
  Teuchos::ArrayRCP<const scalar_type> arr = _x->getData();
  double max_local
    = *std::max_element(arr.get(), arr.get() + arr.size());

  return MPI::max(mpi_comm(), max_local);
}
//-----------------------------------------------------------------------------
double TpetraVector::sum() const
{
  dolfin_assert(!_x.is_null());

  std::vector<int> node_list(local_size());
  Teuchos::ArrayView<int> _node_list(node_list);

  _x->getMap()->getRemoteIndexList(_x->getMap()->getNodeElementList(),
                                   _node_list);

  Teuchos::ArrayRCP<const scalar_type> arr(_x->getData());

  int mpi_rank = _x->getMap()->getComm()->getRank();
  double _sum = 0.0;
  for (std::size_t i = 0; i != local_size(); ++i)
    if (node_list[i] == mpi_rank)
      _sum += arr[i];

  return MPI::sum(mpi_comm(), _sum);
}
//-----------------------------------------------------------------------------
double TpetraVector::sum(const Array<std::size_t>& rows) const
{
  dolfin_assert(!_x.is_null());

  // FIXME

  dolfin_not_implemented();

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
  const TpetraVector& _y = as_type<const TpetraVector>(y);

  _x->elementWiseMultiply(1.0, *_x, *(_y._x), 0.0);

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
const TpetraVector& TpetraVector::operator+= (const GenericVector& y)
{
  axpy(1.0, y);
  return *this;
}
//-----------------------------------------------------------------------------
const TpetraVector& TpetraVector::operator+= (double a)
{
  dolfin_assert(!_x.is_null());

  const std::size_t num_values = local_size();
  for (std::size_t i = 0; i != num_values; ++i)
    _x->sumIntoLocalValue(i, a);

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
const TpetraVector& TpetraVector::operator= (const TpetraVector& v)
{
  // Check that vector lengths are equal
  if (size() != v.size())
  {
    dolfin_error("TpetraVector.cpp",
                 "assign one vector to another",
                 "Vectors must be of the same length when assigning. "
                 "Consider using the copy constructor instead");
  }

  // Check that vector local ranges are equal (relevant in parallel)
  if (local_range() != v.local_range())
  {
    dolfin_error("TpetraVector.cpp",
                 "assign one vector to another",
                 "Vectors must have the same parallel layout when assigning. "
                 "Consider using the copy constructor instead");
  }

  // Check for self-assignment
  if (this != &v)
  {
    // Copy data (local operation)
    dolfin_assert(!v._x.is_null());
    dolfin_assert(!_x.is_null());

    _x->assign(*v._x);
  }

  return *this;
}
//-----------------------------------------------------------------------------
GenericLinearAlgebraFactory& TpetraVector::factory() const
{
  return TpetraFactory::instance();
}
//-----------------------------------------------------------------------------
void TpetraVector::_init(MPI_Comm comm,
                         std::pair<std::size_t, std::size_t> local_range,
                         const std::vector<dolfin::la_index>& local_to_global_map)
{
  if (!_x.is_null())
    error("TpetraVector cannot be initialized more than once.");

  // Make a Trilinos version of the MPI Comm
  Teuchos::RCP<const Teuchos::Comm<int> >
    _comm(new Teuchos::MpiComm<int>(comm));

  // Mapping across processes
  Teuchos::RCP<map_type> _map;
  std::size_t Nlocal = local_range.second - local_range.first;
  std::size_t N = MPI::sum(comm, Nlocal);

  if (local_to_global_map.size()==0)
    _map = Teuchos::rcp(new map_type(N, Nlocal, 0, _comm));
  else
  {
    std::vector<global_ordinal_type> ltmp(local_to_global_map.begin(),
                                          local_to_global_map.end());

    const Teuchos::ArrayView<global_ordinal_type> local_indices(ltmp);
    _map = Teuchos::rcp(new map_type(N, local_indices, 0, _comm));
  }

  // Vector
  _x = Teuchos::rcp(new vector_type(_map));

}
//-----------------------------------------------------------------------------
Teuchos::RCP<vector_type> TpetraVector::vec() const
{
  return _x;
}
//-----------------------------------------------------------------------------
void TpetraVector::mapdump(Teuchos::RCP<const map_type> xmap,
                           const std::string desc)
{
  std::stringstream ss;

  int rank = xmap->getComm()->getRank();
  int m = xmap->getMaxAllGlobalIndex() + 1;
  if (rank == 0)
  {
    ss << xmap->description() << "\n" << desc << "\n---";
    for (int j = 0; j != m ; ++j)
      ss << "-";
    ss << "\n";
  }

  ss << rank << "] ";
  for (int j = 0; j != m ; ++j)
    if (xmap->isNodeGlobalElement(j))
      ss << "X";
    else
      ss << " ";
  ss << "\n";

  const Teuchos::RCP<const Teuchos::MpiComm<int> > _mpi_comm
    = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int> >
    (xmap()->getComm());

  MPI_Comm mpi_comm = *(_mpi_comm->getRawMpiComm());

  std::vector<std::string> out_str;
  MPI::gather(mpi_comm, ss.str(), out_str);

  if (rank == 0)
  {
    for (auto &s: out_str)
      std::cout << s;
  }

}


#endif
