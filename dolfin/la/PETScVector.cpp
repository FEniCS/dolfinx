// Copyright (C) 2004-2012 Johan Hoffman, Johan Jansson and Anders Logg
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
//
// First added:  2004
// Last changed: 2012-08-22

#ifdef HAS_PETSC

#include <cmath>
#include <numeric>
#include <boost/assign/list_of.hpp>
#include <dolfin/common/Timer.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Set.h>
#include <dolfin/log/dolfin_log.h>
#include "PETScVector.h"
#include "uBLASVector.h"
#include "PETScFactory.h"
#include "PETScCuspFactory.h"
#include <dolfin/common/MPI.h>

using namespace dolfin;

const std::map<std::string, NormType> PETScVector::norm_types
= boost::assign::map_list_of("l1",   NORM_1)
  ("l2",   NORM_2)
  ("linf", NORM_INFINITY);

//-----------------------------------------------------------------------------
PETScVector::PETScVector(std::string type, bool use_gpu)
  : _use_gpu(use_gpu)
{
  if (type != "global" && type != "local")
  {
    dolfin_error("PETScVector.cpp",
                 "create PETSc vector",
                 "Unknown vector type (\"%s\")", type.c_str());
  }

#ifndef HAS_PETSC_CUSP
  if (_use_gpu)
  {
    dolfin_error("PETScVector.cpp",
                 "create GPU vector",
                 "PETSc not compiled with Cusp support");
  }
#endif

  // Empty ghost indices vector
  const std::vector<la_index> ghost_indices;

  // Trivial range
  const std::pair<std::size_t, std::size_t> range(0, 0);

  if (type == "global" && dolfin::MPI::num_processes(PETSC_COMM_WORLD) > 1)
    _init(range, ghost_indices, true);
  else
    _init(range, ghost_indices, false);
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(std::size_t N, std::string type, bool use_gpu)
  : _use_gpu(use_gpu)
{
#ifndef HAS_PETSC_CUSP
  if (_use_gpu)
  {
    dolfin_error("PETScVector.cpp",
                 "create GPU vector",
                 "PETSc not compiled with Cusp support");
  }
#endif

  // Empty ghost indices vector
  const std::vector<la_index> ghost_indices;

  if (type == "global")
  {
    // Compute a local range
    const std::pair<std::size_t, std::size_t> range
      = MPI::local_range(MPI_COMM_WORLD, N);

    if (range.first == 0 && range.second == N)
      _init(range, ghost_indices, false);
    else
      _init(range, ghost_indices, true);
  }
  else if (type == "local")
  {
    const std::pair<std::size_t, std::size_t> range(0, N);
    _init(range, ghost_indices, false);
  }
  else
  {
    dolfin_error("PETScVector.cpp",
                 "create PETSc vector",
                 "Unknown vector type (\"%s\")", type.c_str());
  }
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(const GenericSparsityPattern& sparsity_pattern)
  : _use_gpu(false)
{
  std::vector<la_index> ghost_indices;
  resize(sparsity_pattern.local_range(0), ghost_indices);
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(boost::shared_ptr<Vec> x): _x(x), _use_gpu(false)
{
  // Do nothing else
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(const PETScVector& v)
  : _x(new Vec(0), PETScVectorDeleter()), _use_gpu(false)
{
  PetscErrorCode ierr;

  dolfin_assert(v._x);

  // Create new vector
  ierr = VecDuplicate(*(v._x), _x.get());
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecDuplicate");

  // Copy data
  ierr = VecCopy(*(v._x), *_x);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecCopy");

  // Copy ghost data
  ghost_global_to_local = v.ghost_global_to_local;

  // Create ghost view
  x_ghosted.reset(new Vec(0), PETScVectorDeleter());
  if (!ghost_global_to_local.empty())
  {
    ierr = VecGhostGetLocalForm(*_x, x_ghosted.get());
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostGetLocalForm");
  }
}
//-----------------------------------------------------------------------------
PETScVector::~PETScVector()
{
  // Do nothing. The custom shared_ptr deleter takes care of the cleanup.
}
//-----------------------------------------------------------------------------
bool PETScVector::distributed() const
{
  dolfin_assert(_x);

  // Get type
  #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 3
  const VecType petsc_type;
  #else
  VecType petsc_type;
  #endif
  PetscErrorCode ierr = VecGetType(*_x, &petsc_type);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetType");

  // Return type
  bool _distributed = false;
  if (strcmp(petsc_type, VECMPI) == 0)
    _distributed = true;
  else if (strcmp(petsc_type, VECSEQ) == 0)
    _distributed =  false;
  #ifdef HAS_PETSC_CUSP
  // TODO: Uncomment these two lines after implementing MPI Cusp vectors
  //else if (strcmp(petsc_type, VECMPICUSP) == 0)
  //  _distributed = true;
  else if (strcmp(petsc_type, VECSEQCUSP) == 0)
    _distributed = false;
  #endif
  else
  {
    dolfin_error("PETScVector.cpp",
                 "check whether PETSc vector is distributed",
                 "Unknown vector type (\"%s\")", petsc_type);
  }

  return _distributed;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericVector> PETScVector::copy() const
{
  boost::shared_ptr<GenericVector> v(new PETScVector(*this));
  return v;
}
//-----------------------------------------------------------------------------
void PETScVector::resize(std::size_t N)
{
  if (_x && this->size() == N)
    return;

  if (!_x)
  {
    dolfin_error("PETScVector.cpp",
                 "resize PETSc vector",
                 "Vector has not been initialized");
  }

  // Get vector type
  const bool _distributed = distributed();

  // Create vector
  if (_distributed)
  {
    const std::pair<std::size_t, std::size_t> range
      = MPI::local_range(MPI_COMM_WORLD, N);
    resize(range);
  }
  else
  {
    const std::pair<std::size_t, std::size_t> range(0, N);
    resize(range);
  }
}
//-----------------------------------------------------------------------------
void PETScVector::resize(std::pair<std::size_t, std::size_t> range)
{
  // Create empty ghost indices vector
  std::vector<la_index> ghost_indices;
  resize(range, ghost_indices);
}
//-----------------------------------------------------------------------------
void PETScVector::resize(std::pair<std::size_t, std::size_t> range,
                         const std::vector<la_index>& ghost_indices)
{
  // FIXME: Can this check be made robust? Need to avoid parallel lock-up.
  //        Cannot just check size because range may change.
  // Check if resizing is required
  //if (x && (this->local_range().first == range.first && this->local_range().second == range.second))
  //  return;

  // Get type
  const bool _distributed = distributed();

  // Re-initialise vector
  _init(range, ghost_indices, _distributed);
}
//-----------------------------------------------------------------------------
void PETScVector::get_local(std::vector<double>& values) const
{
  dolfin_assert(_x);
  const std::size_t n0 = local_range().first;
  const std::size_t local_size = local_range().second - local_range().first;
  values.resize(local_size);

  if (local_size == 0)
    return;

  std::vector<PetscInt> rows(local_size, n0);
  for (std::size_t i = 0; i < local_size; ++i)
    rows[i] += i;

  PetscErrorCode ierr = VecGetValues(*_x, local_size, rows.data(),
                                     values.data());
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetValues");
}
//-----------------------------------------------------------------------------
void PETScVector::set_local(const std::vector<double>& values)
{
  dolfin_assert(_x);
  const std::size_t n0 = local_range().first;
  const std::size_t local_size = local_range().second - local_range().first;
  if (values.size() != local_size)
  {
    dolfin_error("PETScVector.cpp",
                 "set local values of PETSc vector",
                 "Size of values array is not equal to local vector size");
  }

  if (local_size == 0)
    return;

  // Build array of global indices
  std::vector<PetscInt> rows(local_size, n0);
  for (std::size_t i = 0; i < local_size; ++i)
    rows[i] += i;

  PetscErrorCode ierr = VecSetValues(*_x, local_size, rows.data(),
                                     values.data(), INSERT_VALUES);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecSetValues");
}
//-----------------------------------------------------------------------------
void PETScVector::add_local(const Array<double>& values)
{
  dolfin_assert(_x);
  const std::size_t n0 = local_range().first;
  const std::size_t local_size = local_range().second - local_range().first;
  if (values.size() != local_size)
  {
    dolfin_error("PETScVector.cpp",
                 "add local values to PETSc vector",
                 "Size of values array is not equal to local vector size");
  }

  if (local_size == 0)
    return;

  // Build array of global indices
  std::vector<PetscInt> rows(local_size, n0);
  for (std::size_t i = 0; i < local_size; ++i)
    rows[i] += i;

  PetscErrorCode ierr = VecSetValues(*_x, local_size, rows.data(),
                                     values.data(), ADD_VALUES);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecSetValues");
}
//-----------------------------------------------------------------------------
void PETScVector::get_local(double* block, std::size_t m,
			    const dolfin::la_index* rows) const
{
  dolfin_assert(_x);
  PetscErrorCode ierr;
  PetscInt _m = m;
  const dolfin::la_index* _rows = rows;

  // Handle case that m = 0 (VecGetValues is collective -> must be called be
  //                         all processes)
  if (m == 0)
  {
    _rows = &_m;
    double tmp = 0.0;
    block = &tmp;
  }

  // Use VecGetValues if no ghost points, otherwise check for ghost values
  if (ghost_global_to_local.empty() || m == 0)
  {
    ierr = VecGetValues(*_x, _m, _rows, block);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetValues");
  }
  else
  {
    dolfin_assert(x_ghosted);

    // Get local range
    const PetscInt n0 = local_range().first;
    const PetscInt n1 = local_range().second;
    const PetscInt local_size = n1 - n0;

    // Build list of rows, and get from ghosted vector
    std::vector<PetscInt> local_rows(m);
    for (std::size_t i = 0; i < m; ++i)
    {
      if (rows[i] >= n0 && rows[i] < n1)
        local_rows[i] = rows[i] - n0;
      else
      {
        boost::unordered_map<std::size_t, std::size_t>::const_iterator
          local_index = ghost_global_to_local.find(rows[i]);
        dolfin_assert(local_index != ghost_global_to_local.end());
        local_rows[i] = local_index->second + local_size;
      }
    }

    // Pick values from ghosted vector
    ierr = VecGetValues(*x_ghosted, _m, local_rows.data(), block);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetValues");
  }
}
//-----------------------------------------------------------------------------
void PETScVector::set(const double* block, std::size_t m,
                      const dolfin::la_index* rows)
{
  dolfin_assert(_x);
  if (m == 0)
    return;
  PetscErrorCode ierr = VecSetValues(*_x, m, rows, block, INSERT_VALUES);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecSetValues");
}
//-----------------------------------------------------------------------------
void PETScVector::add(const double* block, std::size_t m,
                      const dolfin::la_index* rows)
{
  dolfin_assert(_x);
  if (m == 0)
    return;
  PetscErrorCode ierr = VecSetValues(*_x, m, rows, block, ADD_VALUES);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecSetValues");
}
//-----------------------------------------------------------------------------
void PETScVector::apply(std::string mode)
{
  Timer timer("Apply (PETScVector)");
  dolfin_assert(_x);
  PetscErrorCode ierr;
  ierr = VecAssemblyBegin(*_x);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecAssemblyBegin");
  ierr = VecAssemblyEnd(*_x);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecAssemblyEnd");
}
//-----------------------------------------------------------------------------
void PETScVector::zero()
{
  dolfin_assert(_x);
  double a = 0.0;
  PetscErrorCode ierr = VecSet(*_x, a);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecSet");
}
//-----------------------------------------------------------------------------
bool PETScVector::empty() const
{
  return size() == 0;
}
//-----------------------------------------------------------------------------
std::size_t PETScVector::size() const
{
  PetscInt n = 0;
  if (_x)
  {
    PetscErrorCode ierr = VecGetSize(*_x, &n);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetSize");
  }
  return n;
}
//-----------------------------------------------------------------------------
std::size_t PETScVector::local_size() const
{
  PetscInt n = 0;
  if (_x)
  {
    PetscErrorCode ierr = VecGetLocalSize(*_x, &n);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetLocalSize");
  }
  return n;
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t> PETScVector::local_range() const
{
  PetscInt n0, n1;
  PetscErrorCode ierr = VecGetOwnershipRange(*_x, &n0, &n1);
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
    // Copy data (local operatrion)
    dolfin_assert(v._x);
    dolfin_assert(_x);
    PetscErrorCode ierr = VecCopy(*(v._x), *_x);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecCopy");
  }
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator= (double a)
{
  dolfin_assert(_x);
  PetscErrorCode ierr = VecSet(*_x, a);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecSet");
  return *this;
}
//-----------------------------------------------------------------------------
void PETScVector::update_ghost_values()
{
  #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 3
  if (dolfin::MPI::num_processes() > 1)
  #endif
  {
    PetscErrorCode ierr;
    ierr = VecGhostUpdateBegin(*_x, INSERT_VALUES, SCATTER_FORWARD);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostUpdateBegin");
    ierr = VecGhostUpdateEnd(*_x, INSERT_VALUES, SCATTER_FORWARD);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostUpdateEnd");
  }
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
  PetscErrorCode ierr = VecShift(*_x, a);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecShift");
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
  PetscErrorCode ierr = VecScale(*_x, a);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScale");
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

  PetscErrorCode ierr = VecPointwiseMult(*_x, *_x, *v._x);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecPointwiseMult");
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
  PetscErrorCode ierr = VecDot(*(_y._x), *_x, &a);
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

  PetscErrorCode ierr = VecAXPY(*_x, a, *(_y._x));
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecAXPY");
}
//-----------------------------------------------------------------------------
void PETScVector::abs()
{
  dolfin_assert(_x);
  VecAbs(*_x);
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
  PetscErrorCode ierr = VecNorm(*_x, norm_types.find(norm_type)->second,
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
  PetscErrorCode ierr = VecMin(*_x, &position, &value);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecMin");
  return value;
}
//-----------------------------------------------------------------------------
double PETScVector::max() const
{
  dolfin_assert(_x);
  double value = 0.0;
  PetscInt position = 0;
  PetscErrorCode ierr = VecMax(*_x, &position, &value);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecMax");
  return value;
}
//-----------------------------------------------------------------------------
double PETScVector::sum() const
{
  dolfin_assert(_x);
  double value = 0.0;
  PetscErrorCode ierr = VecSum(*_x, &value);
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
  const std::size_t num_processes  = MPI::num_processes(MPI_COMM_WORLD);
  const std::size_t process_number = MPI::process_number(MPI_COMM_WORLD);
  for (std::size_t i = 1; i < num_processes; ++i)
  {
    // Receive data from process p - i (i steps to the left), send data to
    // process p + i (i steps to the right)
    const std::size_t source
      = (process_number - i + num_processes) % num_processes;
    const std::size_t dest   = (process_number + i) % num_processes;

    // Send and receive data
    std::vector<std::size_t> received_nonlocal_rows;
    MPI::send_recv(MPI_COMM_WORLD, send_nonlocal_rows.set(), dest,
                   received_nonlocal_rows, source);

    // Add rows which reside on this process
    for (std::size_t j = 0; j < received_nonlocal_rows.size(); ++j)
    {
      if (received_nonlocal_rows[j] >= n0 && received_nonlocal_rows[j] < n1)
        local_rows.insert(received_nonlocal_rows[j]);
    }
  }

  // Get local values
  std::vector<double> local_values(local_rows.size());
  get_local(&local_values[0], local_rows.size(), &local_rows.set()[0]);

  // Compute local sum
  const double local_sum = std::accumulate(local_values.begin(),
                                           local_values.end(), 0.0);

  return MPI::sum(MPI_COMM_WORLD, local_sum);
}
//-----------------------------------------------------------------------------
std::string PETScVector::str(bool verbose) const
{
  if (!_x)
    return "<Uninitialized PETScVector>";

  PetscErrorCode ierr;

  std::stringstream s;
  if (verbose)
  {
    // Get vector type
    #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 3
    const VecType petsc_type;
    #else
    VecType petsc_type;
    #endif
    dolfin_assert(_x);
    ierr = VecGetType(*_x, &petsc_type);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGet");

    if (strcmp(petsc_type, VECSEQ) == 0)
    {
      ierr = VecView(*_x, PETSC_VIEWER_STDOUT_SELF);
      if (ierr != 0) petsc_error(ierr, __FILE__, "VecView");
    }
    else if (strcmp(petsc_type, VECMPI) == 0)
    {
      ierr = VecView(*_x, PETSC_VIEWER_STDOUT_WORLD);
      if (ierr != 0) petsc_error(ierr, __FILE__, "VecView");
    }
    #ifdef HAS_PETSC_CUSP
    else if (strcmp(petsc_type, VECSEQCUSP) == 0)
    {
      ierr = VecView(*_x, PETSC_VIEWER_STDOUT_SELF);
      if (ierr != 0) petsc_error(ierr, __FILE__, "VecView");
    // TODO: Uncomment these two lines after implementing MPI Cusp vectors
    //else if (strcmp(petsc_type, VECMPICUSP) == 0)
    //  VecView(*x, PETSC_VIEWER_STDOUT_WORLD);
    }
    #endif
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

  // Check that y is a local vector
  #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 3
  const VecType petsc_type;
  #else
  VecType petsc_type;
  #endif
  ierr = VecGetType(*(_y.vec()), &petsc_type);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetType");

  #ifndef HAS_PETSC_CUSP
  // If PETSc is configured without Cusp, check only for one sequential type
  if (strcmp(petsc_type, VECSEQ) != 0)
  {
    dolfin_error("PETScVector.cpp",
                 "gather values for PETSc vector",
                 "Values can only be gathered into local vectors");
  }
  #else
  // If PETSc is configured with Cusp, check for both sequential types
  if (strcmp(petsc_type, VECSEQ) != 0 && strcmp(petsc_type, VECSEQCUSP) != 0)
  {
    dolfin_error("PETScVector.cpp",
                 "gather values for PETSc vector",
                 "Values can only be gathered into local vectors");
  }
  #endif

  // Prepare data for index sets (global indices)
  std::vector<PetscInt> global_indices(indices.begin(), indices.end());

  // Prepare data for index sets (local indices)
  const PetscInt n = indices.size();

  // PETSc will bail out if it receives a NULL pointer even though m == 0.
  // Can't return from function since function calls are collective.
  if (n == 0)
    global_indices.resize(1);

  // Create local index sets
  IS from, to;
  ierr = ISCreateGeneral(PETSC_COMM_SELF, n, global_indices.data(),
                         PETSC_COPY_VALUES, &from);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISCreateGeneral");
  ierr = ISCreateStride(PETSC_COMM_SELF, n, 0 , 1, &to);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISCreateStride");

  // Resize vector if required
  y.resize(n);

  // Perform scatter
  VecScatter scatter;
  ierr = VecScatterCreate(*_x, from, *(_y.vec()), to, &scatter);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScatterCreate");
  ierr = VecScatterBegin(scatter, *_x, *(_y.vec()), INSERT_VALUES,
                         SCATTER_FORWARD);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScatterBegin");
  ierr = VecScatterEnd(scatter, *_x, *(_y.vec()), INSERT_VALUES,
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
void PETScVector::gather(std::vector<double>& x, const std::vector<dolfin::la_index>& indices) const
{
  x.resize(indices.size());
  PETScVector y("local");
  gather(y, indices);
  dolfin_assert(y.local_size() == x.size());
  y.get_local(x);
}
//-----------------------------------------------------------------------------
void PETScVector::gather_on_zero(std::vector<double>& x) const
{
  PetscErrorCode ierr;

  if (MPI::process_number(MPI_COMM_WORLD) == 0)
    x.resize(size());
  else
    x.resize(0);

  boost::shared_ptr<Vec> vout(new Vec(0), PETScVectorDeleter());
  VecScatter scatter;
  ierr = VecScatterCreateToZero(*_x, &scatter, vout.get());
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScatterCreateToZero");
  ierr = VecScatterBegin(scatter, *_x, *vout, INSERT_VALUES, SCATTER_FORWARD);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScatterBegin");
  ierr = VecScatterEnd(scatter, *_x, *vout, INSERT_VALUES, SCATTER_FORWARD);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScatterEnd");
  ierr = VecScatterDestroy(&scatter);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScatterDestroy");

  // Wrap PETSc vector
  if (MPI::process_number(MPI_COMM_WORLD) == 0)
  {
    PETScVector _vout(vout);
    _vout.get_local(x);
  }
}
//-----------------------------------------------------------------------------
void PETScVector::_init(std::pair<std::size_t, std::size_t> range,
                        const std::vector<la_index>& ghost_indices,
                        bool distributed)
{
  PetscErrorCode ierr;
  // Create vector
  if (_x && !_x.unique())
  {
    dolfin_error("PETScVector.cpp",
                 "initialize PETSc vector",
                 "More than one object points to the underlying PETSc object");
  }
  _x.reset(new Vec(0), PETScVectorDeleter());

  const std::size_t local_size = range.second - range.first;
  dolfin_assert(range.second >= range.first);

  // Initialize vector, either default or MPI vector
  if (!distributed)
  {
    ierr = VecCreate(PETSC_COMM_SELF, _x.get());
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecCreate");

    // Set type to be either standard or Cusp sequential vector
    if (!_use_gpu)
    {
      ierr = VecSetType(*_x, VECSEQ);
      if (ierr != 0) petsc_error(ierr, __FILE__, "VecSetType");
    }
    #ifdef HAS_PETSC_CUSP
    else
    {
      ierr = VecSetType(*_x, VECSEQCUSP);
      if (ierr != 0) petsc_error(ierr, __FILE__, "VecSetType");
    }
    #endif

    ierr = VecSetSizes(*_x, local_size, PETSC_DECIDE);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecSetSizes");
    ierr = VecSetFromOptions(*_x);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecSetFromOptions");
  }
  else
  {
    if (_use_gpu)
    {
      not_working_in_parallel("Due to limitations in PETSc, "
          "distributed PETSc Cusp vectors");
    }

    // Clear ghost indices map
    ghost_global_to_local.clear();

    // Copy ghost indices
    ierr = VecCreateGhost(PETSC_COMM_WORLD, local_size, PETSC_DECIDE,
                          ghost_indices.size(), ghost_indices.data(), _x.get());
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecCreateGhost");

    // Build global-to-local map for ghost indices
    for (std::size_t i = 0; i < ghost_indices.size(); ++i)
    {
      ghost_global_to_local.insert(std::pair<std::size_t,
                                             std::size_t>(ghost_indices[i], i));
    }

    // Create ghost view
    x_ghosted.reset(new Vec(0), PETScVectorDeleter());
    ierr = VecGhostGetLocalForm(*_x, x_ghosted.get());
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostGetLocalForm");
  }
}
//-----------------------------------------------------------------------------
boost::shared_ptr<Vec> PETScVector::vec() const
{
  return _x;
}
//-----------------------------------------------------------------------------
void PETScVector::reset()
{
  _x.reset();
  x_ghosted.reset();
  ghost_global_to_local.clear();
}
//-----------------------------------------------------------------------------
GenericLinearAlgebraFactory& PETScVector::factory() const
{
  if (!_use_gpu)
    return PETScFactory::instance();
  #ifdef HAS_PETSC_CUSP
  else
    return PETScCuspFactory::instance();
  #endif

  // Return something to keep the compiler happy. Code will never be reached.
  return PETScFactory::instance();
}
//-----------------------------------------------------------------------------
#endif
