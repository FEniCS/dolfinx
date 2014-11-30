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
  _x->putScalar(0.0);
}
//-----------------------------------------------------------------------------
void TpetraVector::apply(std::string mode)
{
  dolfin_assert(!_x.is_null());
  // ?
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
  std::vector<std::size_t> local_to_global_map;
  _init(comm, range, local_to_global_map);
}
//-----------------------------------------------------------------------------
void TpetraVector::init(MPI_Comm comm,
                        std::pair<std::size_t, std::size_t> range)
{
  std::vector<std::size_t> local_to_global_map;
  _init(comm, range, local_to_global_map);
}
//-----------------------------------------------------------------------------
void TpetraVector::init(MPI_Comm comm,
                        std::pair<std::size_t, std::size_t> range,
                        const std::vector<std::size_t>& local_to_global_map,
                        const std::vector<la_index>& ghost_indices)
{
  _init(comm, range, local_to_global_map);
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
  // FXIME: wrong? with ghost entries?
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
  dolfin_assert(!_x.is_null());

  // Make map of global indices
  std::vector<global_ordinal_type> _rows(rows, rows + m);
  const Teuchos::ArrayView<const global_ordinal_type> local_indices(_rows);
  Teuchos::RCP<const Teuchos::Comm<int> > 
    _comm(new Teuchos::MpiComm<int>(MPI_COMM_SELF));
  Teuchos::RCP<const map_type> 
    ymap(new map_type(m, local_indices, 0, _comm));  

  // Create local Vector on map and import to it
  Teuchos::RCP<vector_type> y(new vector_type(ymap));
  const Tpetra::Import<global_ordinal_type> 
    importer(_x->getMap(), y->getMap());
  y->doImport(*_x, importer, Tpetra::INSERT);
 
  // Copy to memory
  Teuchos::ArrayRCP<const scalar_type> arr = y->getData();
  std::copy(arr.get(), arr.get() + m, block);
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
  //  for (std::size_t i = 0; i != num_values; ++i)
  //    _x->replaceLocalValue(i, values[i]);
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

  if (num_values == 0)
    return;

  for (std::size_t i = 0; i != num_values; ++i)
    _x->sumIntoLocalValue(i, values[i]);
}
//-----------------------------------------------------------------------------
void TpetraVector::gather(GenericVector& y,
                          const std::vector<dolfin::la_index>& indices) const
{
  dolfin_assert(!_x.is_null());
  TpetraVector& _y = as_type<TpetraVector>(y);

  std::vector<global_ordinal_type> 
    global_indices(indices.begin(), indices.end());  

  // Prepare data for index sets (local indices)
  const std::size_t n = indices.size();

  if (_y._x.is_null())
    _y.init(MPI_COMM_SELF, n);
  else if (y.size() != n || MPI::size(y.mpi_comm()) != 1)
  {
    dolfin_error("TpetraVector.cpp",
                 "gather vector entries",
                 "Cannot re-initialize gather vector. Must be empty, or have correct size and be a local vector");
  }

  // FIXME: something like
  // _y._x->replaceMap(global_indices);
  
  const Tpetra::Import<global_ordinal_type> 
    importer(_x->getMap(), _y._x->getMap());
  _y._x->doImport(*_x, importer, Tpetra::INSERT);
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
  
  if (MPI::rank(mpi_comm()) == 0)
    v.resize(size());
  else
    v.resize(0);

  TpetraVector y;
  y.init(v.size(), MPI_COMM_SELF);
  
  const Tpetra::Import<global_ordinal_type> 
    importer(_x->getMap(), y._x->getMap());
  y._x->doImport(*_x, importer, Tpetra::INSERT);

  y.get_local(v);
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
    = *std::min_element(arr.get(), arr.get() + local_size());
 
  return MPI::min(mpi_comm(), min_local);
}
//-----------------------------------------------------------------------------
double TpetraVector::max() const
{
  dolfin_assert(!_x.is_null());
  Teuchos::ArrayRCP<const scalar_type> arr = _x->getData();
  double max_local
    = *std::max_element(arr.get(), arr.get() + local_size());
 
  return MPI::max(mpi_comm(), max_local);
}
//-----------------------------------------------------------------------------
double TpetraVector::sum() const
{
  dolfin_assert(!_x.is_null());
  dolfin_not_implemented();
  return 0.0;
}
//-----------------------------------------------------------------------------
double TpetraVector::sum(const Array<std::size_t>& rows) const
{
  dolfin_assert(!_x.is_null());
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
                         const std::vector<std::size_t>& local_to_global_map)
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
#endif
