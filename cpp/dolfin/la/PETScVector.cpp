// Copyright (C) 2004-2016 Johan Hoffman, Johan Jansson, Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PETScVector.h"
#include "SparsityPattern.h"
#include "utils.h"
#include <cmath>
#include <cstddef>
#include <cstring>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/log/log.h>
#include <numeric>

using namespace dolfin;
using namespace dolfin::la;

namespace
{
const std::map<std::string, NormType> norm_types
    = {{"l1", NORM_1}, {"l2", NORM_2}, {"linf", NORM_INFINITY}};
}

#define CHECK_ERROR(NAME)                                                      \
  do                                                                           \
  {                                                                            \
    if (ierr != 0)                                                             \
      petsc_error(ierr, __FILE__, NAME);                                       \
  } while (0)

//-----------------------------------------------------------------------------
PETScVector::PETScVector(MPI_Comm comm) : _x(nullptr)
{
  PetscErrorCode ierr = VecCreate(comm, &_x);
  CHECK_ERROR("VecCreate");
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(Vec x) : _x(x)
{
  // Increase reference count to PETSc object
  PetscObjectReference((PetscObject)_x);
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(const PETScVector& v) : _x(nullptr)
{
  dolfin_assert(v._x);

  // Duplicate vector
  PetscErrorCode ierr;
  ierr = VecDuplicate(v._x, &_x);
  CHECK_ERROR("VecDuplicate");

  // Copy data
  ierr = VecCopy(v._x, _x);
  CHECK_ERROR("VecCopy");

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
void PETScVector::init(std::size_t N)
{
  const auto range = dolfin::MPI::local_range(this->mpi_comm(), N);
  init(range, {}, {}, 1);
}
//-----------------------------------------------------------------------------
void PETScVector::init(std::array<std::int64_t, 2> range)
{
  init(range, {}, {}, 1);
}
//-----------------------------------------------------------------------------
void PETScVector::init(std::array<std::int64_t, 2> range,
                       const std::vector<la_index_t>& local_to_global_map,
                       const std::vector<la_index_t>& ghost_indices,
                       int block_size)
{
  if (!_x)
  {
    log::dolfin_error("PETScVector.h", "initialize vector",
                      "Underlying PETSc Vec has not been initialized");
  }

  PetscErrorCode ierr;

  // Set from PETSc options. This will set the vector type.
  ierr = VecSetFromOptions(_x);
  CHECK_ERROR("VecSetFromOptions");

  // Get local size
  dolfin_assert(range[1] >= range[0]);
  const std::size_t local_size = block_size * (range[1] - range[0]);

  // Set vector size
  ierr = VecSetSizes(_x, local_size, PETSC_DECIDE);
  CHECK_ERROR("VecSetSizes");

  // Set block size
  ierr = VecSetBlockSize(_x, block_size);
  CHECK_ERROR("VecSetBlockSize");

  // Get PETSc Vec type
  VecType vec_type = nullptr;
  ierr = VecGetType(_x, &vec_type);
  CHECK_ERROR("VecGetType");

  // Add ghost points if Vec type is MPI (throw an error if Vec is not
  // VECMPI and ghost entry vector is not empty)
  if (strcmp(vec_type, VECMPI) == 0)
  {
    // Note: is re-creating the vector ok?
    MPI_Comm comm = this->mpi_comm();
    Vec y;
    ierr = VecCreateGhostBlock(comm, block_size, local_size, PETSC_DECIDE,
                               ghost_indices.size(), ghost_indices.data(), &y);
    if (_x)
      VecDestroy(&_x);
    _x = y;

    /*
    // This version has problem when setting the block size
    if (block_size == 1)
      ierr = VecMPISetGhost(_x, ghost_indices.size(), ghost_indices.data());
    else
    {
      std::vector<PetscInt> _ghost_indices(block_size*ghost_indices.size());
      for (std::size_t i = 0; i < ghost_indices.size(); ++i)
        for (int j = 0; j < block_size; ++j)
          _ghost_indices[block_size*i + j] = block_size*ghost_indices[i] + j;

      ierr = VecMPISetGhost(_x, _ghost_indices.size(), _ghost_indices.data());
    }
    */
    CHECK_ERROR("VecMPISetGhost");
  }
  else if (!ghost_indices.empty())
  {
    log::dolfin_error("PETScVector.cpp", "initialize vector",
                      "Sequential PETSc Vec objects cannot have ghost entries");
  }

  // Build local-to-global map
  ISLocalToGlobalMapping petsc_local_to_global;
  if (!local_to_global_map.empty())
  {
    // Create PETSc local-to-global map
    ierr = ISLocalToGlobalMappingCreate(
        PETSC_COMM_SELF, block_size, local_to_global_map.size(),
        local_to_global_map.data(), PETSC_COPY_VALUES, &petsc_local_to_global);
    CHECK_ERROR("ISLocalToGlobalMappingCreate");
  }
  else
  {
    // Fill vector with [i0 + 0, i0 + 1, i0 +2, . . .]
    std::vector<PetscInt> map(local_size);
    std::iota(map.begin(), map.end(), range[0]);

    // Create PETSc local-to-global map
    ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, block_size, map.size(),
                                        map.data(), PETSC_COPY_VALUES,
                                        &petsc_local_to_global);
    CHECK_ERROR("ISLocalToGlobalMappingCreate");
  }

  // Apply local-to-global map to vector
  ierr = VecSetLocalToGlobalMapping(_x, petsc_local_to_global);
  CHECK_ERROR("VecSetLocalToGlobalMapping");

  // Clean-up PETSc local-to-global map
  ierr = ISLocalToGlobalMappingDestroy(&petsc_local_to_global);
  CHECK_ERROR("ISLocalToGlobalMappingDestroy");
}
//-----------------------------------------------------------------------------
std::int64_t PETScVector::size() const
{
  dolfin_assert(_x);
  PetscErrorCode ierr;

  // Return zero if vector type has not been set (Vec has not been
  // initialized)
  VecType vec_type = nullptr;
  ierr = VecGetType(_x, &vec_type);
  if (vec_type == nullptr)
    return 0;
  CHECK_ERROR("VecGetType");

  PetscInt n = 0;
  dolfin_assert(_x);
  ierr = VecGetSize(_x, &n);
  CHECK_ERROR("VecGetSize");

  return n > 0 ? n : 0;
}
//-----------------------------------------------------------------------------
std::size_t PETScVector::local_size() const
{
  dolfin_assert(_x);
  PetscErrorCode ierr;

  // Return zero if vector type has not been set
  VecType vec_type = nullptr;
  ierr = VecGetType(_x, &vec_type);
  if (vec_type == nullptr)
    return 0;
  CHECK_ERROR("VecGetType");

  PetscInt n = 0;
  ierr = VecGetLocalSize(_x, &n);
  CHECK_ERROR("VecGetLocalSize");

  return n;
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> PETScVector::local_range() const
{
  dolfin_assert(_x);

  PetscInt n0, n1;
  PetscErrorCode ierr = VecGetOwnershipRange(_x, &n0, &n1);
  CHECK_ERROR("VecGetOwnershipRange");
  dolfin_assert(n0 <= n1);
  return {{n0, n1}};
}
//-----------------------------------------------------------------------------
void PETScVector::get_local(std::vector<double>& values) const
{
  dolfin_assert(_x);
  const auto _local_range = local_range();
  const std::size_t local_size = _local_range[1] - _local_range[0];
  values.resize(local_size);

  if (local_size == 0)
    return;

  // Get pointer to PETSc vector data
  const PetscScalar* data;
  PetscErrorCode ierr = VecGetArrayRead(_x, &data);
  CHECK_ERROR("VecGetArrayRead");

  // Copy data into vector
  std::copy(data, data + local_size, values.begin());

  // Restore array
  ierr = VecRestoreArrayRead(_x, &data);
  CHECK_ERROR("VecRestoreArrayRead");
}
//-----------------------------------------------------------------------------
void PETScVector::set_local(const std::vector<double>& values)
{
  dolfin_assert(_x);
  const auto _local_range = local_range();
  const std::size_t local_size = _local_range[1] - _local_range[0];
  if (values.size() != local_size)
  {
    log::dolfin_error("PETScVector.cpp", "set local values of PETSc vector",
                      "Size of values array is not equal to local vector size");
  }

  if (local_size == 0)
    return;

  // Build array of local indices
  std::vector<PetscInt> rows(local_size, 0);
  std::iota(rows.begin(), rows.end(), 0);

  PetscErrorCode ierr = VecSetValuesLocal(_x, local_size, rows.data(),
                                          values.data(), INSERT_VALUES);
  CHECK_ERROR("VecSetValuesLocal");
}
//-----------------------------------------------------------------------------
void PETScVector::add_local(const std::vector<double>& values)
{
  dolfin_assert(_x);
  const auto _local_range = local_range();
  const std::size_t local_size = _local_range[1] - _local_range[0];
  if (values.size() != local_size)
  {
    log::dolfin_error("PETScVector.cpp", "add local values to PETSc vector",
                      "Size of values array is not equal to local vector size");
  }

  if (local_size == 0)
    return;

  // Build array of local indices
  std::vector<PetscInt> rows(local_size);
  std::iota(rows.begin(), rows.end(), 0);

  PetscErrorCode ierr = VecSetValuesLocal(_x, local_size, rows.data(),
                                          values.data(), ADD_VALUES);
  CHECK_ERROR("VecSetValuesLocal");
}
//-----------------------------------------------------------------------------
void PETScVector::get_local(double* block, std::size_t m,
                            const dolfin::la_index_t* rows) const
{
  if (m == 0)
    return;

  dolfin_assert(_x);
  PetscErrorCode ierr;

  // Get ghost vector
  Vec xg = nullptr;
  ierr = VecGhostGetLocalForm(_x, &xg);
  CHECK_ERROR("VecGhostGetLocalForm");

  // Use array access if no ghost points, otherwise use VecGetValues
  // on local ghosted form of vector
  if (!xg)
  {
    // Get pointer to PETSc vector data
    const PetscScalar* data;
    ierr = VecGetArrayRead(_x, &data);
    CHECK_ERROR("VecGetArrayRead");

    for (std::size_t i = 0; i < m; ++i)
      block[i] = data[rows[i]];

    // Restore array
    ierr = VecRestoreArrayRead(_x, &data);
    CHECK_ERROR("VecRestoreArrayRead");
  }
  else
  {
    dolfin_assert(xg);
    ierr = VecGetValues(xg, m, rows, block);
    CHECK_ERROR("VecGetValues");

    ierr = VecGhostRestoreLocalForm(_x, &xg);
    CHECK_ERROR("VecGhostRestoreLocalForm");
  }
}
//-----------------------------------------------------------------------------
void PETScVector::get(double* block, std::size_t m,
                      const dolfin::la_index_t* rows) const
{
  if (m == 0)
    return;

  dolfin_assert(_x);
  PetscErrorCode ierr = VecGetValues(_x, m, rows, block);
  CHECK_ERROR("VecGetValues");
}
//-----------------------------------------------------------------------------
void PETScVector::set(const double* block, std::size_t m,
                      const dolfin::la_index_t* rows)
{
  dolfin_assert(_x);
  PetscErrorCode ierr = VecSetValues(_x, m, rows, block, INSERT_VALUES);
  CHECK_ERROR("VecSetValues");
}
//-----------------------------------------------------------------------------
void PETScVector::set_local(const double* block, std::size_t m,
                            const dolfin::la_index_t* rows)
{
  dolfin_assert(_x);
  PetscErrorCode ierr = VecSetValuesLocal(_x, m, rows, block, INSERT_VALUES);
  CHECK_ERROR("VecSetValuesLocal");
}
//-----------------------------------------------------------------------------
void PETScVector::add(const double* block, std::size_t m,
                      const dolfin::la_index_t* rows)
{
  dolfin_assert(_x);
  PetscErrorCode ierr = VecSetValues(_x, m, rows, block, ADD_VALUES);
  CHECK_ERROR("VecSetValues");
}
//-----------------------------------------------------------------------------
void PETScVector::add_local(const double* block, std::size_t m,
                            const dolfin::la_index_t* rows)
{
  dolfin_assert(_x);
  PetscErrorCode ierr = VecSetValuesLocal(_x, m, rows, block, ADD_VALUES);
  CHECK_ERROR("VecSetValuesLocal");
}
//-----------------------------------------------------------------------------
void PETScVector::apply()
{
  common::Timer timer("Apply (PETScVector)");
  dolfin_assert(_x);
  PetscErrorCode ierr;
  ierr = VecAssemblyBegin(_x);
  CHECK_ERROR("VecAssemblyBegin");
  ierr = VecAssemblyEnd(_x);
  CHECK_ERROR("VecAssemblyEnd");

  // Update any ghost values
  update_ghost_values();
}
//-----------------------------------------------------------------------------
MPI_Comm PETScVector::mpi_comm() const
{
  dolfin_assert(_x);
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  PetscErrorCode ierr = PetscObjectGetComm((PetscObject)(_x), &mpi_comm);
  CHECK_ERROR("PetscObjectGetComm");
  return mpi_comm;
}
//-----------------------------------------------------------------------------
void PETScVector::zero()
{
  dolfin_assert(_x);
  double a = 0.0;
  PetscErrorCode ierr = VecSet(_x, a);
  CHECK_ERROR("VecSet");
  this->apply();
}
//-----------------------------------------------------------------------------
bool PETScVector::empty() const { return this->size() == 0; }
//-----------------------------------------------------------------------------
bool PETScVector::owns_index(std::size_t i) const
{
  const auto _local_range = local_range();
  const std::int64_t _i = i;
  return _i >= _local_range[0] && _i < _local_range[1];
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator=(const PETScVector& v)
{
  _x = v._x;
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator=(double a)
{
  dolfin_assert(_x);
  PetscErrorCode ierr = VecSet(_x, a);
  CHECK_ERROR("VecSet");
  apply();
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
  CHECK_ERROR("VecGhostGetLocalForm");

  if (xg)
  {
    ierr = VecGhostUpdateBegin(_x, INSERT_VALUES, SCATTER_FORWARD);
    CHECK_ERROR("VecGhostUpdateBegin");
    ierr = VecGhostUpdateEnd(_x, INSERT_VALUES, SCATTER_FORWARD);
    CHECK_ERROR("VecGhostUpdateEnd");
  }

  ierr = VecGhostRestoreLocalForm(_x, &xg);
  CHECK_ERROR("VecGhostRestoreLocalForm");
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator+=(const PETScVector& x)
{
  axpy(1.0, x);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator+=(double a)
{
  dolfin_assert(_x);
  PetscErrorCode ierr = VecShift(_x, a);
  CHECK_ERROR("VecShift");

  // Update any ghost values
  update_ghost_values();

  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator-=(const PETScVector& x)
{
  axpy(-1.0, x);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator-=(double a)
{
  dolfin_assert(_x);
  (*this) += -a;
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator*=(const double a)
{
  dolfin_assert(_x);
  PetscErrorCode ierr = VecScale(_x, a);
  CHECK_ERROR("VecScale");

  // Update ghost values
  update_ghost_values();

  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator*=(const PETScVector& v)
{
  dolfin_assert(_x);
  dolfin_assert(v._x);
  if (size() != v.size())
  {
    log::dolfin_error("PETScVector.cpp",
                      "perform point-wise multiplication with PETSc vector",
                      "Vectors are not of the same size");
  }

  PetscErrorCode ierr = VecPointwiseMult(_x, _x, v._x);
  CHECK_ERROR("VecPointwiseMult");

  // Update ghost values
  update_ghost_values();

  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator/=(const double a)
{
  dolfin_assert(_x);
  dolfin_assert(a != 0.0);
  const double b = 1.0 / a;
  (*this) *= b;
  return *this;
}
//-----------------------------------------------------------------------------
double PETScVector::dot(const PETScVector& y) const
{
  dolfin_assert(_x);
  dolfin_assert(y._x);
  double a;
  PetscErrorCode ierr = VecDot(y._x, _x, &a);
  CHECK_ERROR("VecDot");
  return a;
}
//-----------------------------------------------------------------------------
void PETScVector::axpy(double a, const PETScVector& y)
{
  dolfin_assert(_x);

  dolfin_assert(y._x);
  if (size() != y.size())
  {
    log::dolfin_error("PETScVector.cpp",
                      "perform axpy operation with PETSc vector",
                      "Vectors are not of the same size");
  }

  PetscErrorCode ierr = VecAXPY(_x, a, y._x);
  CHECK_ERROR("VecAXPY");

  // Update ghost values
  update_ghost_values();
}
//-----------------------------------------------------------------------------
void PETScVector::abs()
{
  dolfin_assert(_x);
  PetscErrorCode ierr = VecAbs(_x);
  CHECK_ERROR("VecAbs");

  // Update ghost values
  update_ghost_values();
}
//-----------------------------------------------------------------------------
double PETScVector::norm(std::string norm_type) const
{
  dolfin_assert(_x);
  if (norm_types.count(norm_type) == 0)
  {
    log::dolfin_error("PETScVector.cpp", "compute norm of PETSc vector",
                      "Unknown norm type (\"%s\")", norm_type.c_str());
  }

  double value = 0.0;
  PetscErrorCode ierr = VecNorm(_x, norm_types.find(norm_type)->second, &value);
  CHECK_ERROR("VecNorm");
  return value;
}
//-----------------------------------------------------------------------------
double PETScVector::min() const
{
  dolfin_assert(_x);
  double value = 0.0;
  PetscInt position = 0;
  PetscErrorCode ierr = VecMin(_x, &position, &value);
  CHECK_ERROR("VecMin");
  return value;
}
//-----------------------------------------------------------------------------
double PETScVector::max() const
{
  dolfin_assert(_x);
  double value = 0.0;
  PetscInt position = 0;
  PetscErrorCode ierr = VecMax(_x, &position, &value);
  CHECK_ERROR("VecMax");
  return value;
}
//-----------------------------------------------------------------------------
double PETScVector::sum() const
{
  dolfin_assert(_x);
  double value = 0.0;
  PetscErrorCode ierr = VecSum(_x, &value);
  CHECK_ERROR("VecSum");
  return value;
}
//-----------------------------------------------------------------------------
std::string PETScVector::str(bool verbose) const
{
  dolfin_assert(_x);
  PetscErrorCode ierr;

  // Check if vector type has not been set
  VecType vec_type = nullptr;
  ierr = VecGetType(_x, &vec_type);
  if (vec_type == nullptr)
    return "<Uninitialized PETScVector>";
  CHECK_ERROR("VecGetType");

  std::stringstream s;
  if (verbose)
  {
    // Get vector type
    VecType petsc_type = nullptr;
    dolfin_assert(_x);
    ierr = VecGetType(_x, &petsc_type);
    CHECK_ERROR("VecGetType");

    if (strcmp(petsc_type, VECSEQ) == 0)
    {
      ierr = VecView(_x, PETSC_VIEWER_STDOUT_SELF);
      CHECK_ERROR("VecView");
    }
    else if (strcmp(petsc_type, VECMPI) == 0)
    {
      ierr = VecView(_x, PETSC_VIEWER_STDOUT_WORLD);
      CHECK_ERROR("VecView");
    }
  }
  else
    s << "<PETScVector of size " << size() << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
void PETScVector::gather(PETScVector& y,
                         const std::vector<dolfin::la_index_t>& indices) const
{
  dolfin_assert(_x);
  PetscErrorCode ierr;

  // Get number of required entries
  const std::int64_t n = indices.size();

  // Check that passed vector is local
  if (MPI::size(y.mpi_comm()) != 1)
  {
    log::dolfin_error("PETScVector.cpp", "gather vector entries",
                      "Gather vector must be a local vector (MPI_COMM_SELF)");
  }

  // Initialize vector if empty
  if (y.empty())
    y.init(n);

  // Check that passed vector has correct size
  if (y.size() != n)
  {
    log::dolfin_error("PETScVector.cpp", "gather vector entries",
                      "Gather vector must be empty or of correct size "
                      "(same as provided indices)");
  }

  // Prepare data for index sets (global indices)
  std::vector<PetscInt> global_indices(indices.begin(), indices.end());

  // PETSc will bail out if it receives a NULL pointer even though m
  // == 0.  Can't return from function since function calls are
  // collective.
  if (n == 0)
    global_indices.resize(1);

  // Create local index sets
  IS from, to;
  ierr = ISCreateGeneral(PETSC_COMM_SELF, n, global_indices.data(),
                         PETSC_COPY_VALUES, &from);
  CHECK_ERROR("ISCreateGeneral");
  ierr = ISCreateStride(PETSC_COMM_SELF, n, 0, 1, &to);
  CHECK_ERROR("ISCreateStride");

  // Perform scatter
  VecScatter scatter;
  ierr = VecScatterCreate(_x, from, y.vec(), to, &scatter);
  CHECK_ERROR("VecScatterCreate");
  ierr = VecScatterBegin(scatter, _x, y.vec(), INSERT_VALUES, SCATTER_FORWARD);
  CHECK_ERROR("VecScatterBegin");
  ierr = VecScatterEnd(scatter, _x, y.vec(), INSERT_VALUES, SCATTER_FORWARD);
  CHECK_ERROR("VecScatterEnd");

  // Clean up
  ierr = VecScatterDestroy(&scatter);
  CHECK_ERROR("VecScatterDestroy");
  ierr = ISDestroy(&from);
  CHECK_ERROR("ISDestroy");
  ierr = ISDestroy(&to);
  CHECK_ERROR("ISDestroy");
}
//-----------------------------------------------------------------------------
void PETScVector::gather(std::vector<double>& x,
                         const std::vector<dolfin::la_index_t>& indices) const
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
  CHECK_ERROR("VecScatterCreateToZero");
  ierr = VecScatterBegin(scatter, _x, vout, INSERT_VALUES, SCATTER_FORWARD);
  CHECK_ERROR("VecScatterBegin");
  ierr = VecScatterEnd(scatter, _x, vout, INSERT_VALUES, SCATTER_FORWARD);
  CHECK_ERROR("VecScatterEnd");
  ierr = VecScatterDestroy(&scatter);
  CHECK_ERROR("VecScatterDestroy");

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
    log::dolfin_error(
        "PETScVector.cpp", "setting PETSc options prefix",
        "Cannot set options prefix since PETSc Vec has not been initialized");
  }

  // Set PETSc options prefix
  PetscErrorCode ierr = VecSetOptionsPrefix(_x, options_prefix.c_str());
  CHECK_ERROR("VecSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string PETScVector::get_options_prefix() const
{
  if (!_x)
  {
    log::dolfin_error(
        "PETScVector.cpp", "get PETSc options prefix",
        "Cannot get options prefix since PETSc Vec has not been initialized");
  }

  const char* prefix = nullptr;
  PetscErrorCode ierr = VecGetOptionsPrefix(_x, &prefix);
  CHECK_ERROR("VecGetOptionsPrefix");
  return std::string(prefix);
}
//-----------------------------------------------------------------------------
void PETScVector::set_from_options()
{
  if (!_x)
  {
    log::dolfin_error("PETScVector.cpp",
                      "call VecSetFromOptions on PETSc Vec object",
                      "Vec object has not been initialized");
  }

  PetscErrorCode ierr = VecSetFromOptions(_x);
  CHECK_ERROR("VecSetFromOptions");
}
//-----------------------------------------------------------------------------
Vec PETScVector::vec() const { return _x; }
//-----------------------------------------------------------------------------
void PETScVector::reset(Vec vec)
{
  dolfin_assert(_x);
  PetscErrorCode ierr;

  // Decrease reference count to old Vec object
  ierr = VecDestroy(&_x);
  CHECK_ERROR("VecDestroy");

  // Store new Vec object and increment reference count
  _x = vec;
  ierr = PetscObjectReference((PetscObject)_x);
  CHECK_ERROR("PetscObjectReference");
}
//-----------------------------------------------------------------------------
