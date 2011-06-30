// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring
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
// Modified by Garth N. Wells, 2008-2010.
//
// First added:  2008-04-21
// Last changed: 2011-05-11

#ifdef HAS_TRILINOS

#include <cmath>
#include <cstring>
#include <numeric>
#include <utility>
#include <boost/scoped_ptr.hpp>

#include <Epetra_FEVector.h>
#include <Epetra_Export.h>
#include <Epetra_Import.h>
#include <Epetra_BlockMap.h>
#include <Epetra_MultiVector.h>
#include <Epetra_MpiComm.h>
#include <Epetra_SerialComm.h>
#include <Epetra_Vector.h>
#include <Epetra_DataAccess.h>

#include <dolfin/common/Array.h>
#include <dolfin/common/Set.h>
#include <dolfin/common/MPI.h>
#include <dolfin/log/dolfin_log.h>
#include "uBLASVector.h"
#include "PETScVector.h"
#include "EpetraFactory.h"
#include "EpetraVector.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(std::string type) : type(type)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(uint N, std::string type) : type(type)
{
  // Create Epetra vector
  resize(N);
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(boost::shared_ptr<Epetra_FEVector> x) : x(x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(const Epetra_BlockMap& map)
{
  x.reset(new Epetra_FEVector(map));
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(const EpetraVector& v) : type(v.type)
{
  *this = v;
}
//-----------------------------------------------------------------------------
EpetraVector::~EpetraVector()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void EpetraVector::resize(uint N)
{
  if (x && this->size() == N)
    return;

  // Create empty ghost vertices vector
  std::vector<uint> ghost_indices;

  if (type == "global")
  {
    const std::pair<uint, uint> range = MPI::local_range(N);
    resize(range, ghost_indices);
  }
  else if (type == "local")
  {
    const std::pair<uint, uint> range(0, N);
    resize(range, ghost_indices);
  }
  else
    error("Epetra vector type unknown.");
}
//-----------------------------------------------------------------------------
void EpetraVector::resize(std::pair<uint, uint> range)
{
  std::vector<uint> ghost_indices;
  resize(range, ghost_indices);
}
//-----------------------------------------------------------------------------
void EpetraVector::resize(std::pair<uint, uint> range,
                          const std::vector<uint>& ghost_indices)
{
  if (x && !x.unique())
    error("Cannot resize EpetraVector. More than one object points to the underlying Epetra object.");

  // Create ghost data structures
  ghost_global_to_local.clear();

  // Pointer to Epetra map
  boost::scoped_ptr<Epetra_BlockMap> epetra_map;

  // Epetra factory and serial communicator
  const EpetraFactory& f = EpetraFactory::instance();
  Epetra_SerialComm serial_comm = f.get_serial_comm();

  // Compute local size
  const uint local_size = range.second - range.first;
  assert(range.second - range.first >= 0);

  // Create vector
  if (type == "local")
  {
    if (ghost_indices.size() != 0)
      error("Serial EpetraVectors do not suppprt ghost points.");

    // Create map
    epetra_map.reset(new Epetra_BlockMap(-1, local_size, 1, 0, serial_comm));
  }
  else
  {
    // Create map
    Epetra_MpiComm mpi_comm = f.get_mpi_comm();
    epetra_map.reset(new Epetra_BlockMap(-1, local_size, 1, 0, mpi_comm));

    // Build global-to-local map for ghost indices
    for (uint i = 0; i < ghost_indices.size(); ++i)
      ghost_global_to_local.insert(std::pair<uint, uint>(ghost_indices[i], i));
  }

  // Create vector
  x.reset(new Epetra_FEVector(*epetra_map));

  // Create local ghost vector
  const int num_ghost_entries = ghost_indices.size();
  const int* ghost_entries = reinterpret_cast<const int*>(&ghost_indices[0]);
  Epetra_BlockMap ghost_map(num_ghost_entries, num_ghost_entries,
                            ghost_entries, 1, 0, serial_comm);
  x_ghost.reset(new Epetra_Vector(ghost_map));
}
//-----------------------------------------------------------------------------
EpetraVector* EpetraVector::copy() const
{
  assert(x);
  return new EpetraVector(*this);
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraVector::size() const
{
  return x ? x->GlobalLength(): 0;
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraVector::local_size() const
{
  return x ? x->MyLength(): 0;
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> EpetraVector::local_range() const
{
  assert(x);
  assert(x->Map().LinearMap());
  const Epetra_BlockMap& map = x->Map();
  return std::make_pair<uint, uint>(map.MinMyGID(), map.MaxMyGID() + 1);
}
//-----------------------------------------------------------------------------
bool EpetraVector::owns_index(uint i) const
{
  return x->Map().MyGID(i);
}
//-----------------------------------------------------------------------------
void EpetraVector::zero()
{
  assert(x);
  const int err = x->PutScalar(0.0);
  //apply("add");
  if (err != 0)
    error("EpetraVector::zero: Did not manage to perform Epetra_Vector::PutScalar.");
}
//-----------------------------------------------------------------------------
void EpetraVector::apply(std::string mode)
{
  assert(x);

  // Special treatement required for values applied using 'set'
  // This would be simpler if we required that only local values (on this process) can be set
  if (MPI::sum(static_cast<uint>(off_process_set_values.size())) > 0)
  {
    // Create communicator
    const EpetraFactory& f = EpetraFactory::instance();
    Epetra_MpiComm mpi_comm = f.get_mpi_comm();
    Epetra_SerialComm serial_comm = f.get_serial_comm();

    std::vector<int> non_local_indices; non_local_indices.reserve(off_process_set_values.size());
    std::vector<double> non_local_values; non_local_values.reserve(off_process_set_values.size());
    boost::unordered_map<uint, double>::const_iterator entry;
    for (entry = off_process_set_values.begin(); entry != off_process_set_values.end(); ++entry)
    {
      non_local_indices.push_back(entry->first);
      non_local_values.push_back(entry->second);
    }

    // Create map for y
    const int* _non_local_indices = reinterpret_cast<const int*>(&non_local_indices[0]);
    Epetra_BlockMap target_map(-1, non_local_indices.size(), _non_local_indices, 1, 0, mpi_comm);

    // Create vector y (view of non_local_values)
    Epetra_Vector y(View, target_map, &non_local_values[0]);

    // Create importer
    Epetra_Import importer(x->Map(), target_map);

    // Import off-process 'set' data
    if (mode == "add")
      x->Import(y, importer, Add);
    else if (mode == "insert")
      x->Import(y, importer, InsertAdd);

    // Clear map of off-process set values
    off_process_set_values.clear();
  }
  else
  {
    int err = 0;
    if (mode == "add")
      err = x->GlobalAssemble(Add);
    else if (mode == "insert")
      err = x->GlobalAssemble(Insert);
    else
      error("Unknown apply mode in EpetraVector::apply");

    if (err != 0)
      error("EpetraVector::apply: Did not manage to perform Epetra_Vector::GlobalAssemble.");
  }
}
//-----------------------------------------------------------------------------
std::string EpetraVector::str(bool verbose) const
{
  assert(x);

  std::stringstream s;
  if (verbose)
  {
    warning("Verbose output for EpetraVector not implemented, calling Epetra Print directly.");
    x->Print(std::cout);
  }
  else
    s << "<EpetraVector of size " << size() << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
void EpetraVector::get_local(Array<double>& values) const
{
  assert(x);
  values.resize(x->MyLength());

  const int err = x->ExtractCopy(values.data().get(), 0);
  if (err!= 0)
    error("EpetraVector::get: Did not manage to perform Epetra_Vector::ExtractCopy.");
}
//-----------------------------------------------------------------------------
void EpetraVector::set_local(const Array<double>& values)
{
  assert(x);
  const uint local_size = x->MyLength();

 if (values.size() != local_size)
    error("EpetraVector::set_local: length of values array is not equal to local vector size.");

  for (uint i = 0; i < local_size; ++i)
    (*x)[0][i] = values[i];
}
//-----------------------------------------------------------------------------
void EpetraVector::add_local(const Array<double>& values)
{
  assert(x);
  const uint local_size = x->MyLength();
  if (values.size() != local_size)
    error("EpetraVector::add_local: length of values array is not equal to local vector size.");

  for (uint i = 0; i < local_size; ++i)
    (*x)[0][i] += values[i];
}
//-----------------------------------------------------------------------------
void EpetraVector::set(const double* block, uint m, const uint* rows)
{
  assert(x);
  const int err = x->ReplaceGlobalValues(m, reinterpret_cast<const int*>(rows),
                                         block, 0);

  if (err != 0)
    error("EpetraVector::set: Did not manage to perform Epetra_Vector::ReplaceGlobalValues.");

  assert(x);
  const Epetra_BlockMap& map = x->Map();
  assert(x->Map().LinearMap());
  const uint n0 = map.MinMyGID();
  const uint n1 = map.MaxMyGID();

  // Set local values, or add to off-process cache
  for (uint i = 0; i < m; ++i)
  {
    if (rows[i] >= n0 && rows[i] <= n1)
      (*x)[0][rows[i] - n0] = block[i];
    else
      off_process_set_values[rows[i]] = block[i];
  }
}
//-----------------------------------------------------------------------------
void EpetraVector::add(const double* block, uint m, const uint* rows)
{
  if (off_process_set_values.size() > 0)
    error("EpetraVector:: must be called between calling EpetraVector::set and EpetraVector::add.");

  assert(x);
  int err = x->SumIntoGlobalValues(m, reinterpret_cast<const int*>(rows),
                                   block, 0);

  if (err != 0)
    error("EpetraVector::add: Did not manage to perform Epetra_Vector::SumIntoGlobalValues.");
}
//-----------------------------------------------------------------------------
void EpetraVector::get_local(double* block, uint m, const uint* rows) const
{
  assert(x);
  const Epetra_BlockMap& map = x->Map();
  assert(x->Map().LinearMap());
  const uint n0 = map.MinMyGID();

  // Get values
  if (ghost_global_to_local.size() == 0)
  {
    for (uint i = 0; i < m; ++i)
      block[i] = (*x)[0][rows[i] - n0];
  }
  else
  {
    assert(x_ghost);
    const uint n1 = map.MaxMyGID();
    const Epetra_BlockMap& ghost_map = x_ghost->Map();
    for (uint i = 0; i < m; ++i)
    {
      if (rows[i] >= n0 && rows[i] <= n1)
        block[i] = (*x)[0][rows[i] - n0];
      else
      {
        // FIXME: Check if look-up in std::map is faster than Epetra_BlockMap::LID
        // Get local index
        const int local_index = ghost_map.LID(rows[i]);
        assert(local_index != -1);

        //boost::unordered_map<uint, uint>::const_iterator _local_index = ghost_global_to_local.find(rows[i]);
        //assert(_local_index != ghost_global_to_local.end());
        //const int local_index = _local_index->second;

        // Get value
        block[i] = (*x_ghost)[local_index];
      }
    }
  }
}
//-----------------------------------------------------------------------------
void EpetraVector::gather(GenericVector& y,
                          const Array<dolfin::uint>& indices) const
{
  assert(x);

  // Down cast to an EpetraVector
  EpetraVector& _y = y.down_cast<EpetraVector>();

  // Create serial communicator
  const EpetraFactory& f = EpetraFactory::instance();
  Epetra_SerialComm serial_comm = f.get_serial_comm();

  // Create map for y
  const int* _indices = reinterpret_cast<const int*>(indices.data().get());
  Epetra_BlockMap target_map(indices.size(), indices.size(), _indices, 1, 0, serial_comm);

  // Reset vector y
  _y.reset(target_map);
  assert(_y.vec());

  // Create importer
  Epetra_Import importer(target_map, x->Map());

  // Import values into y
  _y.vec()->Import(*x, importer, Insert);
}
//-----------------------------------------------------------------------------
void EpetraVector::gather(Array<double>& x, const Array<uint>& indices) const
{
  const uint _size = indices.size();
  x.resize(_size);
  assert(x.size() == _size);

  // Gather values into a vector
  EpetraVector y;
  gather(y, indices);

  assert(y.size() == _size);
  const Epetra_FEVector& _y = *(y.vec());

  // Copy values into x
  for (uint i = 0; i < _size; ++i)
    x[i] = (_y)[0][i];
}
//-----------------------------------------------------------------------------
void EpetraVector::gather_on_zero(Array<double>& x) const
{
  // FIXME: Is there an Epetra function for this?

  Array<uint> indices(0);
  if (MPI::process_number() == 0)
  {
    indices.resize(size());
    for (uint i = 0; i < size(); ++i)
      indices[i] = i;
  }

  gather(x, indices);
}
//-----------------------------------------------------------------------------
void EpetraVector::reset(const Epetra_BlockMap& map)
{
  // Clear ghost data
  x_ghost.reset();
  ghost_global_to_local.clear();
  off_process_set_values.clear();

  x.reset(new Epetra_FEVector(map));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<Epetra_FEVector> EpetraVector::vec() const
{
  return x;
}
//-----------------------------------------------------------------------------
double EpetraVector::inner(const GenericVector& y) const
{
  assert(x);

  const EpetraVector& v = y.down_cast<EpetraVector>();
  if (!v.x)
    error("Given vector is not initialized.");

  double a;
  const int err = x->Dot(*(v.x), &a);
  if (err!= 0)
    error("EpetraVector::inner: Did not manage to perform Epetra_Vector::Dot.");

  return a;
}
//-----------------------------------------------------------------------------
void EpetraVector::axpy(double a, const GenericVector& y)
{
  assert(x);

  const EpetraVector& _y = y.down_cast<EpetraVector>();
  if (!_y.x)
    error("Given vector is not initialized.");

  if (size() != _y.size())
    error("The vectors must be of the same size.");

  const int err = x->Update(a, *(_y.vec()), 1.0);
  if (err != 0)
    error("EpetraVector::axpy: Did not manage to perform Epetra_Vector::Update.");
}
//-----------------------------------------------------------------------------
void EpetraVector::abs()
{
  assert(x);
  x->Abs(*x);
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& EpetraVector::factory() const
{
  return EpetraFactory::instance();
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator= (const GenericVector& v)
{
  *this = v.down_cast<EpetraVector>();
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator= (double a)
{
  assert(x);
  x->PutScalar(a);
  return *this;
}
//-----------------------------------------------------------------------------
void EpetraVector::update_ghost_values()
{
  assert(x);
  assert(x_ghost);
  assert(x_ghost->MyLength() == (int) ghost_global_to_local.size());

  // Create importer
  Epetra_Import importer(x_ghost->Map(), x->Map());

  // Import into ghost vector
  x_ghost->Import(*x, importer, Insert);
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator= (const EpetraVector& v)
{
  // FIXME: Epetra assignment operator leads to an errror. Must vectors have
  //        the same size for assigenment to work?

  assert(v.x);
  if (this != &v)
  {
    // Copy vector
    x.reset(new Epetra_FEVector(*(v.x)));

    // Copy ghost data
    if (v.x_ghost)
      x_ghost.reset(new Epetra_Vector(*(v.x_ghost)));
    ghost_global_to_local = v.ghost_global_to_local;
  }
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator+= (const GenericVector& y)
{
  assert(x);
  axpy(1.0, y);
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator-= (const GenericVector& y)
{
  assert(x);
  axpy(-1.0, y);
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator*= (double a)
{
  assert(x);
  const int err = x->Scale(a);
  if (err!= 0)
    error("EpetraVector::operator*=: Did not manage to perform Epetra_Vector::Scale.");
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator*= (const GenericVector& y)
{
  assert(x);
  const EpetraVector& v = y.down_cast<EpetraVector>();

  if (!v.x)
    error("Given vector is not initialized.");

  if (size() != v.size())
    error("The vectors must be of the same size.");

  const int err = x->Multiply(1.0, *x, *v.x, 0.0);
  if (err!= 0)
    error("EpetraVector::operator*=: Did not manage to perform Epetra_Vector::Multiply.");
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator/=(double a)
{
  *this *= 1.0/a;
  return *this;
}
//-----------------------------------------------------------------------------
double EpetraVector::norm(std::string norm_type) const
{
  assert(x);

  double value = 0.0;
  int err = 0;
  if (norm_type == "l1")
    err = x->Norm1(&value);
  else if (norm_type == "l2")
    err = x->Norm2(&value);
  else
    err = x->NormInf(&value);

  if (err != 0)
    error("EpetraVector::norm: Did not manage to compute the norm.");
  return value;
}
//-----------------------------------------------------------------------------
double EpetraVector::min() const
{
  assert(x);
  double value = 0.0;
  const int err = x->MinValue(&value);
  if (err!= 0)
    error("EpetraVector::min: Did not manage to perform Epetra_Vector::MinValue.");
  return value;
}
//-----------------------------------------------------------------------------
double EpetraVector::max() const
{
  assert(x);
  double value = 0.0;
  const int err = x->MaxValue(&value);
  if (err != 0)
    error("EpetraVector::min: Did not manage to perform Epetra_Vector::MinValue.");
  return value;
}
//-----------------------------------------------------------------------------
double EpetraVector::sum() const
{
  assert(x);
  const uint local_size = x->MyLength();

  // Get local values
  Array<double> x_local(local_size);
  get_local(x_local);

  // Compute local sum
  double local_sum = 0.0;
  for (uint i = 0; i < local_size; ++i)
    local_sum += x_local[i];

  // Compute global sum
  double global_sum = 0.0;
  x->Comm().SumAll(&local_sum, &global_sum, 1);

  return global_sum;
}
//-----------------------------------------------------------------------------
double EpetraVector::sum(const Array<uint>& rows) const
{
  assert(x);
  const uint n0 = local_range().first;
  const uint n1 = local_range().second;

  // Build sets of local and nonlocal entries
  Set<uint> local_rows;
  Set<uint> nonlocal_rows;
  for (uint i = 0; i < rows.size(); ++i)
  {
    if (rows[i] >= n0 && rows[i] < n1)
      local_rows.insert(rows[i]);
    else
      nonlocal_rows.insert(rows[i]);
  }

  // Send nonlocal row indices to other processes
  const uint num_processes  = MPI::num_processes();
  const uint process_number = MPI::process_number();
  for (uint i = 1; i < num_processes; ++i)
  {
    // Receive data from process p - i (i steps to the left), send data to
    // process p + i (i steps to the right)
    const uint source = (process_number - i + num_processes) % num_processes;
    const uint dest   = (process_number + i) % num_processes;

    // Size of send and receive data
    uint send_buffer_size = nonlocal_rows.size();
    uint recv_buffer_size = 0;
    MPI::send_recv(&send_buffer_size, 1, dest, &recv_buffer_size, 1, source);

    // Send and receive data
    std::vector<uint> received_nonlocal_rows(recv_buffer_size);
    MPI::send_recv(&(nonlocal_rows.set())[0], send_buffer_size, dest,
                   &received_nonlocal_rows[0], recv_buffer_size, source);

    // Add rows which reside on this process
    for (uint j = 0; j < received_nonlocal_rows.size(); ++j)
    {
      if (received_nonlocal_rows[j] >= n0 && received_nonlocal_rows[j] < n1)
        local_rows.insert(received_nonlocal_rows[j]);
    }
  }

  // Compute local sum
  double local_sum = 0.0;
  for (uint i = 0; i < local_rows.size(); ++i)
    local_sum += (*x)[0][local_rows[i] - n0];

  // Compute global sum
  double global_sum = 0.0;
  x->Comm().SumAll(&local_sum, &global_sum, 1);

  return global_sum;
}
//-----------------------------------------------------------------------------
#endif
