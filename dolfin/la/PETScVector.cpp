// Copyright (C) 2004-2016 Johan Hoffman, Johan Jansson, Anders Logg
// and Garth N. Wells
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
// Modified by Garth N. Wells 2005-2010
// Modified by Martin Sandve Alnes 2008
// Modified by Johannes Ring 2011.
// Modified by Fredrik Valdmanis 2011-2012

#ifdef HAS_PETSC

#include <cmath>
#include <cstring>
#include <numeric>
#include <dolfin/common/Timer.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/MPI.h>
#include <dolfin/log/log.h>
#include "SparsityPattern.h"
#include "PETScVector.h"
#include "PETScFactory.h"

using namespace dolfin;

const std::map<std::string, NormType> PETScVector::norm_types
= { {"l1",   NORM_1}, {"l2",   NORM_2},  {"linf", NORM_INFINITY} };

//-----------------------------------------------------------------------------
PETScVector::PETScVector() : PETScVector(MPI_COMM_WORLD)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(MPI_Comm comm) : _x(NULL)
{
  PetscErrorCode ierr = VecCreate(comm, &_x);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecCreate");
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(MPI_Comm comm, std::size_t N) : PETScVector(comm)
{
  // Compute a local range and initialise vector
  const auto range = dolfin::MPI::local_range(comm, N);
  _init(range, {}, {});
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(const SparsityPattern& sparsity_pattern)
  : PETScVector(sparsity_pattern.mpi_comm())
{
  _init(sparsity_pattern.local_range(0), {}, {});
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(Vec x): _x(x)
{
  // Increase reference count to PETSc object
  PetscObjectReference((PetscObject)_x);
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(const PETScVector& v) : _x(nullptr)
{
  dolfin_assert(v._x);
  PetscErrorCode ierr;

  // Create new vector
  ierr = VecDuplicate(v._x, &_x);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecDuplicate");

  // Copy data
  ierr = VecCopy(v._x, _x);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecCopy");

  // Update ghost values
  update_ghost_values();
}
//-----------------------------------------------------------------------------
PETScVector::~PETScVector()
{
  if (_x)
    VecDestroy(&_x);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericVector> PETScVector::copy() const
{
  return std::make_shared<PETScVector>(*this);
}
//-----------------------------------------------------------------------------
void PETScVector::init(MPI_Comm comm, std::size_t N)
{
  const auto range = dolfin::MPI::local_range(comm, N);
  _init(range, {}, {});
}
//-----------------------------------------------------------------------------
void PETScVector::init(MPI_Comm comm,
                       std::pair<std::size_t, std::size_t> range)
{
  _init(range, {}, {});
}
//-----------------------------------------------------------------------------
void PETScVector::init(MPI_Comm comm,
                       std::pair<std::size_t, std::size_t> range,
                       const std::vector<std::size_t>& local_to_global_map,
                       const std::vector<la_index>& ghost_indices)
{
  // Initialise vector
  _init(range, local_to_global_map, ghost_indices);
}
//-----------------------------------------------------------------------------
void PETScVector::get_local(std::vector<double>& values) const
{
  dolfin_assert(_x);
  const std::size_t local_size = local_range().second - local_range().first;
  values.resize(local_size);

  if (local_size == 0)
    return;

  // Get pointer to PETSc vector data
  const PetscScalar* data;
  PetscErrorCode ierr = VecGetArrayRead(_x, &data);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetArrayRead");

  // Copy data into vector
  std::copy(data, data + local_size, values.begin());

  // Restore array
  ierr = VecRestoreArrayRead(_x, &data);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecRestoreArrayRead");
}
//-----------------------------------------------------------------------------
void PETScVector::set_local(const std::vector<double>& values)
{
  dolfin_assert(_x);
  const std::size_t local_size = local_range().second - local_range().first;
  if (values.size() != local_size)
  {
    dolfin_error("PETScVector.cpp",
                 "set local values of PETSc vector",
                 "Size of values array is not equal to local vector size");
  }

  if (local_size == 0)
    return;

  // Build array of local indices
  std::vector<PetscInt> rows(local_size, 0);
  std::iota(rows.begin(), rows.end(), 0);

  PetscErrorCode ierr = VecSetValuesLocal(_x, local_size, rows.data(),
                                          values.data(), INSERT_VALUES);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecSetValues");
}
//-----------------------------------------------------------------------------
void PETScVector::add_local(const Array<double>& values)
{
  dolfin_assert(_x);
  const std::size_t local_size = local_range().second - local_range().first;
  if (values.size() != local_size)
  {
    dolfin_error("PETScVector.cpp",
                 "add local values to PETSc vector",
                 "Size of values array is not equal to local vector size");
  }

  if (local_size == 0)
    return;

  // Build array of local indices
  std::vector<PetscInt> rows(local_size);
  std::iota(rows.begin(), rows.end(), 0);

  PetscErrorCode ierr = VecSetValuesLocal(_x, local_size, rows.data(),
                                          values.data(), ADD_VALUES);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecSetValuesLocal");
}
//-----------------------------------------------------------------------------
void PETScVector::get_local(double* block, std::size_t m,
			    const dolfin::la_index* rows) const
{
  if (m == 0)
    return;

  dolfin_assert(_x);
  PetscErrorCode ierr;

  // Get ghost vector
  Vec xg = NULL;
  ierr = VecGhostGetLocalForm(_x, &xg);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostGetLocalForm");

  // Use array access if no ghost points, otherwise use VecGetValues
  // on local ghosted form of vector
  if (!xg)
  {
    // Get pointer to PETSc vector data
    const PetscScalar* data;
    PetscErrorCode ierr = VecGetArrayRead(_x, &data);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetArrayRead");

    for (std::size_t i = 0; i < m; ++i)
      block[i] = data[rows[i]];

    // Restore array
    ierr = VecRestoreArrayRead(_x, &data);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecRestoreArrayRead");
  }
  else
  {
    dolfin_assert(xg);
    ierr = VecGetValues(xg, m, rows, block);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetValues");

    ierr = VecGhostRestoreLocalForm(_x, &xg);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostRestoreLocalForm");
  }
}
//-----------------------------------------------------------------------------
void PETScVector::get(double* block, std::size_t m,
                      const dolfin::la_index* rows) const
{
  if (m == 0)
    return;

  dolfin_assert(_x);
  PetscErrorCode ierr;
  ierr = VecGetValues(_x, m, rows, block);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetValues");
}
//-----------------------------------------------------------------------------
void PETScVector::set(const double* block, std::size_t m,
                      const dolfin::la_index* rows)
{
  dolfin_assert(_x);
  PetscErrorCode ierr = VecSetValues(_x, m, rows, block, INSERT_VALUES);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecSetValues");
}
//-----------------------------------------------------------------------------
void PETScVector::set_local(const double* block, std::size_t m,
                            const dolfin::la_index* rows)
{
  dolfin_assert(_x);
  PetscErrorCode ierr = VecSetValuesLocal(_x, m, rows, block, INSERT_VALUES);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecSetValuesLocal");
}
//-----------------------------------------------------------------------------
void PETScVector::add(const double* block, std::size_t m,
                      const dolfin::la_index* rows)
{
  dolfin_assert(_x);
  PetscErrorCode ierr = VecSetValues(_x, m, rows, block, ADD_VALUES);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecSetValues");
}
//-----------------------------------------------------------------------------
void PETScVector::add_local(const double* block, std::size_t m,
                            const dolfin::la_index* rows)
{
  dolfin_assert(_x);
  PetscErrorCode ierr = VecSetValuesLocal(_x, m, rows, block, ADD_VALUES);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecSetValuesLocal");
}
//-----------------------------------------------------------------------------
void PETScVector::apply(std::string mode)
{
  Timer timer("Apply (PETScVector)");
  dolfin_assert(_x);
  PetscErrorCode ierr;
  ierr = VecAssemblyBegin(_x);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecAssemblyBegin");
  ierr = VecAssemblyEnd(_x);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecAssemblyEnd");

  // Update any ghost values
  update_ghost_values();
}
//-----------------------------------------------------------------------------
MPI_Comm PETScVector::mpi_comm() const
{
  dolfin_assert(_x);
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  PetscObjectGetComm((PetscObject)(_x), &mpi_comm);
  return mpi_comm;
}
//-----------------------------------------------------------------------------
void PETScVector::zero()
{
  dolfin_assert(_x);
  double a = 0.0;
  PetscErrorCode ierr = VecSet(_x, a);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecSet");
  this->apply("insert");
}
//-----------------------------------------------------------------------------
bool PETScVector::empty() const
{
  return this->size() == 0;
}
//-----------------------------------------------------------------------------
std::size_t PETScVector::size() const
{
  // Return zero if vector type has not been set
  VecType vec_type = NULL;
  VecGetType(_x, &vec_type);
  if (vec_type == NULL)
    return 0;

  PetscInt n = 0;
  dolfin_assert(_x);
  PetscErrorCode ierr = VecGetSize(_x, &n);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetSize");

  return n;
}
//-----------------------------------------------------------------------------
std::size_t PETScVector::local_size() const
{
  dolfin_assert(_x);

  // Return zero if vector type has not been set
  VecType vec_type;
  VecGetType(_x, &vec_type);
  if (vec_type == NULL)
    return 0;

  PetscInt n = 0;
  PetscErrorCode ierr = VecGetLocalSize(_x, &n);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetLocalSize");

  return n;
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t> PETScVector::local_range() const
{
  dolfin_assert(_x);

  PetscInt n0, n1;
  PetscErrorCode ierr = VecGetOwnershipRange(_x, &n0, &n1);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetOwnershipRange");
  dolfin_assert(n0 <= n1);
  return std::make_pair(n0, n1);
}
//-----------------------------------------------------------------------------
bool PETScVector::owns_index(std::size_t i) const
{
  if (i >= local_range().first && i < local_range().second)
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
const GenericVector& PETScVector::operator= (const GenericVector& v)
{
  *this = as_type<const PETScVector>(v);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator= (const PETScVector& v)
{
  // Check that vector lengths are equal
  if (size() != v.size())
  {
    dolfin_error("PETScVector.cpp",
                 "assign one vector to another",
                 "Vectors must be of the same length when assigning. "
                 "Consider using the copy constructor instead");
  }

  // Check that vector local ranges are equal (relevant in parallel)
  if (local_range() != v.local_range())
  {
    dolfin_error("PETScVector.cpp",
                 "assign one vector to another",
                 "Vectors must have the same parallel layout when assigning. "
                 "Consider using the copy constructor instead");
  }

  // Check for self-assignment
  if (this != &v)
  {
    // Copy data (local operation)
    dolfin_assert(v._x);
    dolfin_assert(_x);
    PetscErrorCode ierr = VecCopy(v._x, _x);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecCopy");

    // Update ghost values
    update_ghost_values();
  }
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator= (double a)
{
  dolfin_assert(_x);
  PetscErrorCode ierr = VecSet(_x, a);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecSet");
  apply("insert");
  return *this;
}
//-----------------------------------------------------------------------------
void PETScVector::update_ghost_values()
{
  dolfin_assert(_x);
  PetscErrorCode ierr;

  // Check of vector is ghosted
  Vec xg;
  ierr = VecGhostGetLocalForm(_x, &xg);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostGetLocalForm");

  // If ghosted, update
  if (xg)
  {
    ierr = VecGhostUpdateBegin(_x, INSERT_VALUES, SCATTER_FORWARD);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostUpdateBegin");
    ierr = VecGhostUpdateEnd(_x, INSERT_VALUES, SCATTER_FORWARD);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostUpdateEnd");
  }

  ierr = VecGhostRestoreLocalForm(_x, &xg);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostRestoreLocalForm");
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator+= (const GenericVector& x)
{
  axpy(1.0, x);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator+= (double a)
{
  dolfin_assert(_x);
  PetscErrorCode ierr = VecShift(_x, a);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecShift");

  // Update any ghost values
  update_ghost_values();

  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator-= (const GenericVector& x)
{
  axpy(-1.0, x);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator-= (double a)
{
  dolfin_assert(_x);
  (*this) += -a;
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator*= (const double a)
{
  dolfin_assert(_x);
  PetscErrorCode ierr = VecScale(_x, a);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScale");

  // Update ghost values
  update_ghost_values();

  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator*= (const GenericVector& y)
{
  dolfin_assert(_x);
  const PETScVector& v = as_type<const PETScVector>(y);
  dolfin_assert(v._x);
  if (size() != v.size())
  {
    dolfin_error("PETScVector.cpp",
                 "perform point-wise multiplication with PETSc vector",
                 "Vectors are not of the same size");
  }

  PetscErrorCode ierr = VecPointwiseMult(_x, _x, v._x);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecPointwiseMult");

  // Update ghost values
  update_ghost_values();

  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator/= (const double a)
{
  dolfin_assert(_x);
  dolfin_assert(a != 0.0);
  const double b = 1.0/a;
  (*this) *= b;
  return *this;
}
//-----------------------------------------------------------------------------
double PETScVector::inner(const GenericVector& y) const
{
  dolfin_assert(_x);
  const PETScVector& _y = as_type<const PETScVector>(y);
  dolfin_assert(_y._x);
  double a;
  PetscErrorCode ierr = VecDot(_y._x, _x, &a);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecDot");
  return a;
}
//-----------------------------------------------------------------------------
void PETScVector::axpy(double a, const GenericVector& y)
{
  dolfin_assert(_x);

  const PETScVector& _y = as_type<const PETScVector>(y);
  dolfin_assert(_y._x);
  if (size() != _y.size())
  {
    dolfin_error("PETScVector.cpp",
                 "perform axpy operation with PETSc vector",
                 "Vectors are not of the same size");
  }

  PetscErrorCode ierr = VecAXPY(_x, a, _y._x);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecAXPY");

  // Update ghost values
  update_ghost_values();
}
//-----------------------------------------------------------------------------
void PETScVector::abs()
{
  dolfin_assert(_x);
  VecAbs(_x);

  // Update ghost values
  update_ghost_values();
}
//-----------------------------------------------------------------------------
double PETScVector::norm(std::string norm_type) const
{
  dolfin_assert(_x);
  if (norm_types.count(norm_type) == 0)
  {
    dolfin_error("PETScVector.cpp",
                 "compute norm of PETSc vector",
                 "Unknown norm type (\"%s\")", norm_type.c_str());
  }

  double value = 0.0;
  PetscErrorCode ierr = VecNorm(_x, norm_types.find(norm_type)->second,
                                &value);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecNorm");
  return value;
}
//-----------------------------------------------------------------------------
double PETScVector::min() const
{
  dolfin_assert(_x);
  double value = 0.0;
  PetscInt position = 0;
  PetscErrorCode ierr = VecMin(_x, &position, &value);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecMin");
  return value;
}
//-----------------------------------------------------------------------------
double PETScVector::max() const
{
  dolfin_assert(_x);
  double value = 0.0;
  PetscInt position = 0;
  PetscErrorCode ierr = VecMax(_x, &position, &value);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecMax");
  return value;
}
//-----------------------------------------------------------------------------
double PETScVector::sum() const
{
  dolfin_assert(_x);
  double value = 0.0;
  PetscErrorCode ierr = VecSum(_x, &value);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecSum");
  return value;
}
//-----------------------------------------------------------------------------
double PETScVector::sum(const Array<std::size_t>& rows) const
{
  dolfin_assert(_x);
  const std::size_t n0 = local_range().first;
  const std::size_t n1 = local_range().second;

  // Build sets of local and nonlocal entries
  Set<PetscInt> local_rows;
  Set<std::size_t> send_nonlocal_rows;
  for (std::size_t i = 0; i < rows.size(); ++i)
  {
    if (rows[i] >= n0 && rows[i] < n1)
      local_rows.insert(rows[i]);
    else
      send_nonlocal_rows.insert(rows[i]);
  }

  // Send nonlocal rows indices to other processes
  const std::size_t num_processes  = dolfin::MPI::size(mpi_comm());
  const std::size_t process_number = dolfin::MPI::rank(mpi_comm());
  for (std::size_t i = 1; i < num_processes; ++i)
  {
    // Receive data from process p - i (i steps to the left), send
    // data to process p + i (i steps to the right)
    const std::size_t source
      = (process_number - i + num_processes) % num_processes;
    const std::size_t dest = (process_number + i) % num_processes;

    // Send and receive data
    std::vector<std::size_t> received_nonlocal_rows;
    dolfin::MPI::send_recv(mpi_comm(), send_nonlocal_rows.set(), dest,
                           received_nonlocal_rows, source);

    // Add rows which reside on this process
    for (std::size_t j = 0; j < received_nonlocal_rows.size(); ++j)
    {
      if (received_nonlocal_rows[j] >= n0 && received_nonlocal_rows[j] < n1)
        local_rows.insert(received_nonlocal_rows[j]);
    }
  }

  // Get local values (using global indices)
  std::vector<double> local_values(local_rows.size());
  get(local_values.data(), local_rows.size(), &local_rows.set()[0]);

  // Compute local sum
  const double local_sum = std::accumulate(local_values.begin(),
                                           local_values.end(), 0.0);

  return dolfin::MPI::sum(mpi_comm(), local_sum);
}
//-----------------------------------------------------------------------------
std::string PETScVector::str(bool verbose) const
{
  dolfin_assert(_x);

  // Check if vector type has not been set
  VecType vec_type = NULL;
  VecGetType(_x, &vec_type);
  if (vec_type == NULL)
    return "<Uninitialized PETScVector>";

  PetscErrorCode ierr;
  std::stringstream s;
  if (verbose)
  {
    // Get vector type
    VecType petsc_type;
    dolfin_assert(_x);
    ierr = VecGetType(_x, &petsc_type);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGet");

    if (strcmp(petsc_type, VECSEQ) == 0)
    {
      ierr = VecView(_x, PETSC_VIEWER_STDOUT_SELF);
      if (ierr != 0) petsc_error(ierr, __FILE__, "VecView");
    }
    else if (strcmp(petsc_type, VECMPI) == 0)
    {
      ierr = VecView(_x, PETSC_VIEWER_STDOUT_WORLD);
      if (ierr != 0) petsc_error(ierr, __FILE__, "VecView");
    }
  }
  else
    s << "<PETScVector of size " << size() << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
void PETScVector::gather(GenericVector& y,
                         const std::vector<dolfin::la_index>& indices) const
{
  dolfin_assert(_x);
  PetscErrorCode ierr;

  // Down cast to a PETScVector
  PETScVector& _y = as_type<PETScVector>(y);

  // Prepare data for index sets (global indices)
  std::vector<PetscInt> global_indices(indices.begin(), indices.end());

  // Prepare data for index sets (local indices)
  const std::size_t n = indices.size();

  if (_y.empty())
  {
    // Initialise vector and make local
    y.init(MPI_COMM_SELF, n);
  }
  else if (y.size() != n || dolfin::MPI::size(y.mpi_comm()))
  {
    dolfin_error("PETScVector.cpp",
                 "gather vector entries",
                 "Cannot re-initialize gather vector. Must be empty, or have correct size and be a local vector");
  }


  // PETSc will bail out if it receives a NULL pointer even though m
  // == 0.  Can't return from function since function calls are
  // collective.
  if (n == 0)
    global_indices.resize(1);

  // Create local index sets
  IS from, to;
  ierr = ISCreateGeneral(PETSC_COMM_SELF, n, global_indices.data(),
                         PETSC_COPY_VALUES, &from);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISCreateGeneral");
  ierr = ISCreateStride(PETSC_COMM_SELF, n, 0 , 1, &to);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISCreateStride");


  // Perform scatter
  VecScatter scatter;
  ierr = VecScatterCreate(_x, from, _y.vec(), to, &scatter);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScatterCreate");
  ierr = VecScatterBegin(scatter, _x, _y.vec(), INSERT_VALUES,
                         SCATTER_FORWARD);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScatterBegin");
  ierr = VecScatterEnd(scatter, _x, _y.vec(), INSERT_VALUES,
                       SCATTER_FORWARD);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScatterEnd");

  // Clean up
  ierr = VecScatterDestroy(&scatter);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScatterDestroy");
  ierr = ISDestroy(&from);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISDestroy");
  ierr = ISDestroy(&to);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISDestroy");
}
//-----------------------------------------------------------------------------
void PETScVector::gather(std::vector<double>& x,
                         const std::vector<dolfin::la_index>& indices) const
{
  x.resize(indices.size());
  PETScVector y(PETSC_COMM_SELF);
  gather(y, indices);
  dolfin_assert(y.local_size() == x.size());
  y.get_local(x);
}
//-----------------------------------------------------------------------------
void PETScVector::gather_on_zero(std::vector<double>& x) const
{
  PetscErrorCode ierr;

  if (dolfin::MPI::rank(mpi_comm()) == 0)
    x.resize(size());
  else
    x.resize(0);

  dolfin_assert(_x);
  Vec vout;
  VecScatter scatter;
  ierr = VecScatterCreateToZero(_x, &scatter, &vout);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScatterCreateToZero");
  ierr = VecScatterBegin(scatter, _x, vout, INSERT_VALUES, SCATTER_FORWARD);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScatterBegin");
  ierr = VecScatterEnd(scatter, _x, vout, INSERT_VALUES, SCATTER_FORWARD);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScatterEnd");
  ierr = VecScatterDestroy(&scatter);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScatterDestroy");

  // Wrap PETSc vector
  if (dolfin::MPI::rank(mpi_comm()) == 0)
  {
    PETScVector _vout(vout);
    _vout.get_local(x);
  }
}
//-----------------------------------------------------------------------------
void PETScVector::set_options_prefix(std::string options_prefix)
{
  if (!_x)
  {
    dolfin_error("PETScVector.cpp",
                 "setting PETSc options prefix",
                 "Cannot set options prefix since PETSc Vec has not been initialized");
  }

  // Set PETSc options prefix
  PetscErrorCode ierr = VecSetOptionsPrefix(_x, options_prefix.c_str());
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string PETScVector::get_options_prefix() const
{
  if (!_x)
  {
    dolfin_error("PETScVector.cpp",
                 "get PETSc options prefix",
                 "Cannot get options prefix since PETSc Vec has not been initialized");
  }

  const char* prefix = NULL;
  VecGetOptionsPrefix(_x, &prefix);
  return std::string(prefix);
}
//-----------------------------------------------------------------------------
Vec PETScVector::vec() const
{
  return _x;
}
//-----------------------------------------------------------------------------
GenericLinearAlgebraFactory& PETScVector::factory() const
{
  return PETScFactory::instance();
}
//-----------------------------------------------------------------------------
void PETScVector::_init(std::pair<std::size_t, std::size_t> range,
                        const std::vector<std::size_t>& local_to_global_map,
                        const std::vector<la_index>& ghost_indices)
{
  if (!_x)
  {
    dolfin_error("PETScVector.h",
                 "initialize vector",
                 "Underlying PETSc Vec has not been intialised");
  }

  PetscErrorCode ierr;

  // Set from PETSc options. This will set the vector type.
  VecSetFromOptions(_x);

  // Get local size
  const std::size_t local_size = range.second - range.first;
  dolfin_assert(range.second >= range.first);

  // Set vector size
  VecSetSizes(_x, local_size, PETSC_DECIDE);

  // Get PETSc Vec type
  VecType vec_type;
  VecGetType(_x, &vec_type);

  // Add ghost points if Vec type is MPI (throw an error if Vec is not
  // VECMPI and ghost entry vector is not empty)
  if (strcmp(vec_type, VECMPI) == 0)
  {
    ierr = VecMPISetGhost(_x, ghost_indices.size(), ghost_indices.data());
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecCreateGhost");
  }
  else if (!ghost_indices.empty())
  {
    dolfin_error("PETScVector.cpp",
                 "initialize vector",
                 "Sequential PETSc Vec objects cannot have ghost entries");
  }

  // Build local-to-global map
  std::vector<PetscInt> _map;
  if (!local_to_global_map.empty())
  {
    // Copy data to get correct PETSc integer type
    _map = std::vector<PetscInt>(local_to_global_map.begin(),
                                 local_to_global_map.end());
  }
  else
  {
    // Fill vector with [i0 + 0, i0 + 1, i0 +2, . . .]
    const std::size_t size = range.second - range.first;
    _map.assign(size, range.first);
    std::iota(_map.begin(), _map.end(), range.first);
  }

  // Create PETSc local-to-global map
  ISLocalToGlobalMapping petsc_local_to_global;
  ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, 1, _map.size(), _map.data(),
                               PETSC_COPY_VALUES, &petsc_local_to_global);

  // Apply local-to-global map to vector
  VecSetLocalToGlobalMapping(_x, petsc_local_to_global);

  // Clean-up PETSc local-to-global map
  ISLocalToGlobalMappingDestroy(&petsc_local_to_global);
}
//-----------------------------------------------------------------------------

#endif
