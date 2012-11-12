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
// Modified by Garth N. Wells 2008-2010
// Modified by Anders Logg 2011-2012
//
// First added:  2008-04-21
// Last changed: 2012-08-22

#ifdef HAS_TRILINOS

// Included here to avoid a C++ problem with some MPI implementations
#include <dolfin/common/MPI.h>

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

#include <dolfin/common/Timer.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/Set.h>
#include <dolfin/common/MPI.h>
#include <dolfin/log/dolfin_log.h>
#include "uBLASVector.h"
#include "PETScVector.h"
#include "EpetraVector.h"
#include "EpetraFactory.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(std::string type) : type(type)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(std::size_t N, std::string type) : type(type)
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
  // Copy Epetra vector
  dolfin_assert(v.x);
  x.reset(new Epetra_FEVector(*(v.x)));

  // Copy ghost data
  if (v.x_ghost)
    x_ghost.reset(new Epetra_Vector(*(v.x_ghost)));
  ghost_global_to_local = v.ghost_global_to_local;
}
//-----------------------------------------------------------------------------
EpetraVector::~EpetraVector()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericVector> EpetraVector::copy() const
{
  dolfin_assert(x);
  boost::shared_ptr<GenericVector> y(new EpetraVector(*this));
  return y;
}
//-----------------------------------------------------------------------------
void EpetraVector::resize(std::size_t N)
{
  if (x && this->size() == N)
    return;

  // Create empty ghost vertices vector
  std::vector<std::size_t> ghost_indices;

  if (type == "global")
  {
    const std::pair<std::size_t, std::size_t> range = MPI::local_range(N);
    resize(range, ghost_indices);
  }
  else if (type == "local")
  {
    const std::pair<std::size_t, std::size_t> range(0, N);
    resize(range, ghost_indices);
  }
  else
  {
    dolfin_error("EpetraVector.cpp",
                 "resize Epetra vector",
                 "Unknown vector type (\"%s\")", type.c_str());
  }
}
//-----------------------------------------------------------------------------
void EpetraVector::resize(std::pair<std::size_t, std::size_t> range)
{
  std::vector<std::size_t> ghost_indices;
  resize(range, ghost_indices);
}
//-----------------------------------------------------------------------------
void EpetraVector::resize(std::pair<std::size_t, std::size_t> range,
                          const std::vector<std::size_t>& ghost_indices)
{
  if (x && !x.unique())
  {
    dolfin_error("EpetraVector.cpp",
                 "resize Epetra vector",
                 "More than one object points to the underlying Epetra object");
  }

  // Create ghost data structures
  ghost_global_to_local.clear();

  // Pointer to Epetra map
  boost::scoped_ptr<Epetra_BlockMap> epetra_map;

  // Epetra factory and serial communicator
  EpetraFactory& f = EpetraFactory::instance();
  Epetra_SerialComm serial_comm = f.get_serial_comm();

  // Compute local size
  const std::size_t local_size = range.second - range.first;
  dolfin_assert(int (range.second - range.first) >= 0);

  // Create vector
  if (type == "local")
  {
    if (!ghost_indices.empty())
    {
      dolfin_error("EpetraVector.cpp",
                   "resize Epetra vector",
                   "Serial EpetraVectors do not support ghost points");
    }

    // Create map
    epetra_map.reset(new Epetra_BlockMap(-1, local_size, 1, 0, serial_comm));
  }
  else
  {
    // Create map
    Epetra_MpiComm mpi_comm = f.get_mpi_comm();
    epetra_map.reset(new Epetra_BlockMap(-1, local_size, 1, 0, mpi_comm));

    // Build global-to-local map for ghost indices
    for (std::size_t i = 0; i < ghost_indices.size(); ++i)
      ghost_global_to_local.insert(std::pair<std::size_t, std::size_t>(ghost_indices[i], i));
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
bool EpetraVector::empty() const
{
  return x ? x->GlobalLength() == 0: true;
}
//-----------------------------------------------------------------------------
std::size_t EpetraVector::size() const
{
  return x ? x->GlobalLength(): 0;
}
//-----------------------------------------------------------------------------
std::size_t EpetraVector::local_size() const
{
  return x ? x->MyLength(): 0;
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t> EpetraVector::local_range() const
{
  dolfin_assert(x);
  dolfin_assert(x->Map().LinearMap());
  const Epetra_BlockMap& map = x->Map();
  return std::make_pair<std::size_t, std::size_t>(map.MinMyGID(), map.MaxMyGID() + 1);
}
//-----------------------------------------------------------------------------
bool EpetraVector::owns_index(std::size_t i) const
{
  return x->Map().MyGID((int) i);
}
//-----------------------------------------------------------------------------
void EpetraVector::zero()
{
  Timer("Apply (vector)");
  dolfin_assert(x);
  const int err = x->PutScalar(0.0);
  //apply("add");
  if (err != 0)
  {
    dolfin_error("EpetraVector.cpp",
                 "zero Epetra vector",
                 "Did not manage to perform Epetra_Vector::PutScalar");
  }
}
//-----------------------------------------------------------------------------
void EpetraVector::apply(std::string mode)
{
  dolfin_assert(x);

  // Special treatement required for values applied using 'set'
  // because Epetra_FEVector::ReplaceGlobalValues does not behave well.
  // This would be simpler if we required that only local values (on
  // this process) can be set
  int err = 0;
  if (MPI::sum(static_cast<std::size_t>(off_process_set_values.size())) > 0)
  {
    // Create communicator
    EpetraFactory& f = EpetraFactory::instance();
    Epetra_MpiComm mpi_comm = f.get_mpi_comm();
    Epetra_SerialComm serial_comm = f.get_serial_comm();

    std::vector<int> non_local_indices; non_local_indices.reserve(off_process_set_values.size());
    std::vector<double> non_local_values; non_local_values.reserve(off_process_set_values.size());
    boost::unordered_map<std::size_t, double>::const_iterator entry;
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
      x->Import(y, importer, Insert);

    err = x->GlobalAssemble(Add);
  }
  else
  {
    if (mode == "add")
      err = x->GlobalAssemble(Add);
    else if (mode == "insert")
    {
      // This is 'Add' because we take of data that is set in the previous
      // code block. Epetra behaviour is very poor w.r.t 'set' - probably
      // need to keep track of state set/add internally.
      err = x->GlobalAssemble(Add);
    }
    else
    {
      dolfin_error("EpetraVector.cpp",
                   "apply changes to Epetra vector",
                   "Unknown apply mode (\"%s\")", mode.c_str());
    }
  }

  // Check that call to GlobalAssemble was successful
  if (err != 0)
  {
    dolfin_error("EpetraVector.cpp",
                 "apply changes to Epetra vector",
                 "Did not manage to perform Epetra_Vector::GlobalAssemble");
  }

  // Clear map of off-process set values
  off_process_set_values.clear();
}
//-----------------------------------------------------------------------------
std::string EpetraVector::str(bool verbose) const
{
  if (!x)
    return "<Uninitialized EpetraVector>";

  std::stringstream s;
  if (verbose)
  {
    dolfin_assert(x);
    x->Print(std::cout);
  }
  else
    s << "<EpetraVector of size " << size() << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
void EpetraVector::get_local(std::vector<double>& values) const
{
  if (!x)
  {
    values.clear();
    return;
  }

  values.resize(x->MyLength());

  const int err = x->ExtractCopy(values.data(), 0);
  if (err!= 0)
  {
    dolfin_error("EpetraVector.cpp",
                 "access local values from Epetra vector",
                 "Did not manage to perform Epetra_Vector::ExtractCopy");
  }
}
//-----------------------------------------------------------------------------
void EpetraVector::set_local(const std::vector<double>& values)
{
  dolfin_assert(x);
  const std::size_t local_size = x->MyLength();

  if (values.size() != local_size)
  {
    dolfin_error("EpetraVector.cpp",
                 "set local values of Epetra vector",
                 "Size of values array is not equal to local vector size");
  }

  for (std::size_t i = 0; i < local_size; ++i)
    (*x)[0][i] = values[i];
}
//-----------------------------------------------------------------------------
void EpetraVector::add_local(const Array<double>& values)
{
  dolfin_assert(x);
  const std::size_t local_size = x->MyLength();
  if (values.size() != local_size)
  {
    dolfin_error("EpetraVector.cpp",
                 "add local values to Epetra vector",
                 "Size of values array is not equal to local vector size");
  }

  for (std::size_t i = 0; i < local_size; ++i)
    (*x)[0][i] += values[i];
}
//-----------------------------------------------------------------------------
void EpetraVector::set(const double* block, std::size_t m, const std::size_t* rows)
{
  dolfin_assert(x);
  /*
  for (std::size_t i = 0; i < m; ++i)
  {
	  //const int local_row = x->Map().LID(rows[i]);
    //if (local_row == -1)
    {
      const int err = x->ReplaceGlobalValues(1, (int*) &rows[i], &block[i], 0);
      if (err != 0)
      {
        dolfin_error("EpetraVector.cpp",
                     "set block of values for Epetra vector",
                     "Did not manage to perform Epetra_Vector::ReplaceGlobalValues");
      }
    }
    //else
    //  (*x)[0][local_row] = block[i];
  }
  */

  const Epetra_BlockMap& map = x->Map();
  dolfin_assert(x->Map().LinearMap());
  const std::size_t n0 = map.MinMyGID();
  const std::size_t n1 = map.MaxMyGID();

  // Set local values, or add to off-process cache for later communication
  for (std::size_t i = 0; i < m; ++i)
  {
    if (rows[i] >= n0 && rows[i] <= n1)
      (*x)[0][rows[i] - n0] = block[i];
    else
      off_process_set_values[rows[i]] = block[i];
  }
}
//-----------------------------------------------------------------------------
void EpetraVector::add(const double* block, std::size_t m, const std::size_t* rows)
{
  if (!off_process_set_values.empty())
  {
    dolfin_error("EpetraVector.cpp",
                 "add block of values to Epetra vector",
                 "apply() must be called between calling EpetraVector::set and EpetraVector::add");
  }

  dolfin_assert(x);
  int err = x->SumIntoGlobalValues(m, reinterpret_cast<const int*>(rows),
                                   block, 0);

  if (err != 0)
  {
    dolfin_error("EpetraVector.cpp",
                 "add block of values to Epetra vector",
                 "Did not manage to perform Epetra_Vector::SumIntoGlobalValues");
  }
}
//-----------------------------------------------------------------------------
void EpetraVector::get_local(double* block, std::size_t m, const std::size_t* rows) const
{
  dolfin_assert(x);
  const Epetra_BlockMap& map = x->Map();
  dolfin_assert(x->Map().LinearMap());
  const std::size_t n0 = map.MinMyGID();

  // Get values
  if (ghost_global_to_local.empty())
  {
    for (std::size_t i = 0; i < m; ++i)
      block[i] = (*x)[0][rows[i] - n0];
  }
  else
  {
    dolfin_assert(x_ghost);
    const std::size_t n1 = map.MaxMyGID();
    const Epetra_BlockMap& ghost_map = x_ghost->Map();
    for (std::size_t i = 0; i < m; ++i)
    {
      if (rows[i] >= n0 && rows[i] <= n1)
        block[i] = (*x)[0][rows[i] - n0];
      else
      {
        // FIXME: Check if look-up in std::map is faster than Epetra_BlockMap::LID
        // Get local index
        const int local_index = ghost_map.LID((int) rows[i]);
        dolfin_assert(local_index != -1);

        //boost::unordered_map<std::size_t, std::size_t>::const_iterator _local_index = ghost_global_to_local.find(rows[i]);
        //dolfin_assert(_local_index != ghost_global_to_local.end());
        //const int local_index = _local_index->second;

        // Get value
        block[i] = (*x_ghost)[local_index];
      }
    }
  }
}
//-----------------------------------------------------------------------------
void EpetraVector::gather(GenericVector& y,
                          const std::vector<std::size_t>& indices) const
{
  dolfin_assert(x);

  // Down cast to an EpetraVector
  EpetraVector& _y = as_type<EpetraVector>(y);

  // Create serial communicator
  EpetraFactory& f = EpetraFactory::instance();
  Epetra_SerialComm serial_comm = f.get_serial_comm();

  // Create map for y
  const int* _indices = reinterpret_cast<const int*>(&indices[0]);
  Epetra_BlockMap target_map(indices.size(), indices.size(), _indices, 1, 0, serial_comm);

  // Reset vector y
  _y.reset(target_map);
  dolfin_assert(_y.vec());

  // Create importer
  Epetra_Import importer(target_map, x->Map());

  // Import values into y
  _y.vec()->Import(*x, importer, Insert);
}
//-----------------------------------------------------------------------------
void EpetraVector::gather(std::vector<double>& x, const std::vector<std::size_t>& indices) const
{
  const std::size_t _size = indices.size();
  x.resize(_size);
  dolfin_assert(x.size() == _size);

  // Gather values into a vector
  EpetraVector y;
  gather(y, indices);

  dolfin_assert(y.size() == _size);
  const Epetra_FEVector& _y = *(y.vec());

  // Copy values into x
  for (std::size_t i = 0; i < _size; ++i)
    x[i] = (_y)[0][i];
}
//-----------------------------------------------------------------------------
void EpetraVector::gather_on_zero(std::vector<double>& x) const
{
  // FIXME: Is there an Epetra function for this?

  std::vector<std::size_t> indices(0);
  if (MPI::process_number() == 0)
  {
    indices.resize(size());
    for (std::size_t i = 0; i < size(); ++i)
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
  dolfin_assert(x);

  const EpetraVector& v = as_type<const EpetraVector>(y);
  if (!v.x)
  {
    dolfin_error("EpetraVector.cpp",
                 "compute inner product with Epetra vector",
                 "Given vector is not initialized");
  }

  double a;
  const int err = x->Dot(*(v.x), &a);
  if (err!= 0)
  {
    dolfin_error("EpetraVector.cpp",
                 "compute inner product with Epetra vector",
                 "Did not manage to perform Epetra_Vector::Dot");
  }

  return a;
}
//-----------------------------------------------------------------------------
void EpetraVector::axpy(double a, const GenericVector& y)
{
  dolfin_assert(x);

  const EpetraVector& _y = as_type<const EpetraVector>(y);
  if (!_y.x)
  {
    dolfin_error("EpetraVector.cpp",
                 "perform axpy operation with Epetra vector",
                 "Given vector is not initialized");
  }

  if (size() != _y.size())
  {
    dolfin_error("EpetraVector.cpp",
                 "perform axpy operation with Epetra vector",
                 "Vectors are not of the same size");
  }

  const int err = x->Update(a, *(_y.vec()), 1.0);
  if (err != 0)
  {
    dolfin_error("EpetraVector.cpp",
                 "perform axpy operation with Epetra vector",
                 "Did not manage to perform Epetra_Vector::Update");
  }
}
//-----------------------------------------------------------------------------
void EpetraVector::abs()
{
  dolfin_assert(x);
  x->Abs(*x);
}
//-----------------------------------------------------------------------------
GenericLinearAlgebraFactory& EpetraVector::factory() const
{
  return EpetraFactory::instance();
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator= (const GenericVector& v)
{
  *this = as_type<const EpetraVector>(v);
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator= (double a)
{
  dolfin_assert(x);
  x->PutScalar(a);
  return *this;
}
//-----------------------------------------------------------------------------
void EpetraVector::update_ghost_values()
{
  dolfin_assert(x);
  dolfin_assert(x_ghost);
  dolfin_assert(x_ghost->MyLength() == (int) ghost_global_to_local.size());

  // Create importer
  Epetra_Import importer(x_ghost->Map(), x->Map());

  // Import into ghost vector
  x_ghost->Import(*x, importer, Insert);
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator= (const EpetraVector& v)
{
  // Check that vector lengths are equal
  if (size() != v.size())
  {
    dolfin_error("EpetraVector.cpp",
                 "assign one vector to another",
                 "Vectors must be of the same length when assigning. "
                 "Consider using the copy constructor instead");
  }

  // Check that maps (parallel layout) are the same
  dolfin_assert(x);
  dolfin_assert(v.x);
  if (!x->Map().SameAs(v.x->Map()))
  {
    dolfin_error("EpetraVector.cpp",
                 "assign one vector to another",
                 "Vectors must have the same parallel layout when assigning. "
                 "Consider using the copy constructor instead");
  }

  // Assign values
  *x = *v.x;

  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator+= (const GenericVector& y)
{
  dolfin_assert(x);
  axpy(1.0, y);
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator+= (double a)
{
  dolfin_assert(x);
  Epetra_FEVector y(*x);
  y.PutScalar(a);
  x->Update(1.0, y, 1.0);
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator-= (const GenericVector& y)
{
  dolfin_assert(x);
  axpy(-1.0, y);
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator-= (double a)
{
  dolfin_assert(x);
  Epetra_FEVector y(*x);
  y.PutScalar(-a);
  x->Update(1.0, y, 1.0);
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator*= (double a)
{
  dolfin_assert(x);
  const int err = x->Scale(a);
  if (err!= 0)
  {
    dolfin_error("EpetraVector.cpp",
                 "multiply Epetra vector by scalar",
                 "Did not manage to perform Epetra_Vector::Scale");
  }
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator*= (const GenericVector& y)
{
  dolfin_assert(x);
  const EpetraVector& v = as_type<const EpetraVector>(y);

  if (!v.x)
  {
    dolfin_error("EpetraVector.cpp",
                 "perform point-wise multiplication with Epetra vector",
                 "Given vector is not initialized");
  }

  if (size() != v.size())
  {
    dolfin_error("EpetraVector.cpp",
                 "perform point-wise multiplication with Epetra vector",
                 "Vectors are not of the same size");
  }

  const int err = x->Multiply(1.0, *x, *v.x, 0.0);
  if (err!= 0)
  {
    dolfin_error("EpetraVector.cpp",
                 "perform point-wise multiplication with Epetra vector",
                 "Did not manage to perform Epetra_Vector::Multiply");
  }

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
  dolfin_assert(x);

  double value = 0.0;
  int err = 0;
  if (norm_type == "l1")
    err = x->Norm1(&value);
  else if (norm_type == "l2")
    err = x->Norm2(&value);
  else
    err = x->NormInf(&value);

  if (err != 0)
  {
    dolfin_error("EpetraVector.cpp",
                 "compute norm of Epetra vector",
                 "Did not manage to perform Epetra_vector::Norm");
  }

  return value;
}
//-----------------------------------------------------------------------------
double EpetraVector::min() const
{
  dolfin_assert(x);
  double value = 0.0;
  const int err = x->MinValue(&value);
  if (err!= 0)
  {
    dolfin_error("EpetraVector.cpp",
                 "compute minimum value of Epetra vector",
                 "Did not manage to perform Epetra_Vector::MinValue");
  }

  return value;
}
//-----------------------------------------------------------------------------
double EpetraVector::max() const
{
  dolfin_assert(x);
  double value = 0.0;
  const int err = x->MaxValue(&value);
  if (err != 0)
  {
    dolfin_error("EpetraVector.cpp",
                 "compute maximum value of Epetra vector",
                 "Did not manage to perform Epetra_Vector::MinValue");
  }

  return value;
}
//-----------------------------------------------------------------------------
double EpetraVector::sum() const
{
  dolfin_assert(x);
  const std::size_t local_size = x->MyLength();

  // Get local values
  std::vector<double> x_local;
  get_local(x_local);

  // Compute local sum
  double local_sum = 0.0;
  for (std::size_t i = 0; i < local_size; ++i)
    local_sum += x_local[i];

  // Compute global sum
  double global_sum = 0.0;
  x->Comm().SumAll(&local_sum, &global_sum, 1);

  return global_sum;
}
//-----------------------------------------------------------------------------
double EpetraVector::sum(const Array<std::size_t>& rows) const
{
  dolfin_assert(x);
  const std::size_t n0 = local_range().first;
  const std::size_t n1 = local_range().second;

  // Build sets of local and nonlocal entries
  Set<std::size_t> local_rows;
  Set<std::size_t> nonlocal_rows;
  for (std::size_t i = 0; i < rows.size(); ++i)
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

    // Send and receive data
    std::vector<std::size_t> received_nonlocal_rows;
    MPI::send_recv(nonlocal_rows.set(), dest, received_nonlocal_rows, source);

    // Add rows which reside on this process
    for (std::size_t j = 0; j < received_nonlocal_rows.size(); ++j)
    {
      if (received_nonlocal_rows[j] >= n0 && received_nonlocal_rows[j] < n1)
        local_rows.insert(received_nonlocal_rows[j]);
    }
  }

  // Compute local sum
  double local_sum = 0.0;
  for (std::size_t i = 0; i < local_rows.size(); ++i)
    local_sum += (*x)[0][local_rows[i] - n0];

  // Compute global sum
  double global_sum = 0.0;
  x->Comm().SumAll(&local_sum, &global_sum, 1);

  return global_sum;
}
//-----------------------------------------------------------------------------
#endif
