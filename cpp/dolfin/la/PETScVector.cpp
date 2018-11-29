// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and Garth
// N. Wells
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
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <numeric>

using namespace dolfin;
using namespace dolfin::la;

#define CHECK_ERROR(NAME)                                                      \
  do                                                                           \
  {                                                                            \
    if (ierr != 0)                                                             \
      petsc_error(ierr, __FILE__, NAME);                                       \
  } while (0)

//-----------------------------------------------------------------------------
PETScVector::PETScVector(const common::IndexMap& map)
    : PETScVector(map.mpi_comm(), map.local_range(), map.ghosts(),
                  map.block_size())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(
    MPI_Comm comm, std::array<std::int64_t, 2> range,
    const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& ghost_indices,
    int block_size)
    : _x(nullptr)
{
  PetscErrorCode ierr;

  // Get local size
  assert(range[1] >= range[0]);
  const std::size_t local_size = range[1] - range[0];

  ierr = VecCreateGhostBlock(comm, block_size, block_size * local_size,
                             PETSC_DECIDE, ghost_indices.size(),
                             ghost_indices.data(), &_x);
  CHECK_ERROR("VecCreateGhostBlock");
  assert(_x);

  // Set from PETSc options. This will set the vector type.
  // ierr = VecSetFromOptions(_x);
  // CHECK_ERROR("VecSetFromOptions");

  // NOTE: shouldn't need to do this, but there appears to be an issue
  // with PETSc
  // (https://lists.mcs.anl.gov/pipermail/petsc-dev/2018-May/022963.html)
  // Set local-to-global map
  std::vector<PetscInt> l2g(local_size + ghost_indices.size());
  std::iota(l2g.begin(), l2g.begin() + local_size, range[0]);
  std::copy(ghost_indices.data(), ghost_indices.data() + ghost_indices.size(),
            l2g.begin() + local_size);
  ISLocalToGlobalMapping petsc_local_to_global;
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, block_size, l2g.size(),
                                      l2g.data(), PETSC_COPY_VALUES,
                                      &petsc_local_to_global);
  CHECK_ERROR("ISLocalToGlobalMappingCreate");
  ierr = VecSetLocalToGlobalMapping(_x, petsc_local_to_global);
  CHECK_ERROR("VecSetLocalToGlobalMapping");
  ierr = ISLocalToGlobalMappingDestroy(&petsc_local_to_global);
  CHECK_ERROR("ISLocalToGlobalMappingDestroy");

  // Debug output
  // if (MPI::rank(comm) == 1)
  // {
  //   std::cout << "block size: " << block_size << std::endl;
  //   std::cout << "local size: " << local_size << std::endl;
  //   std::cout << "ghost size: " << ghost_indices.size() << std::endl;
  //   for (std::size_t i = 0; i < ghost_indices.size(); ++i)
  //     std::cout << "  ghost i: " << ghost_indices[i] << std::endl;
  //   ISLocalToGlobalMapping mapping;
  //   VecGetLocalToGlobalMapping(_x, &mapping);
  //   ISLocalToGlobalMappingView(mapping, PETSC_VIEWER_STDOUT_SELF);
  // }
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector() : _x(nullptr) {}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(Vec x) : _x(x)
{
  // Increase reference count to PETSc object
  assert(x);
  PetscObjectReference((PetscObject)_x);
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(const PETScVector& v) : _x(nullptr)
{
  PetscErrorCode ierr;
  assert(v._x);
  ierr = VecDuplicate(v._x, &_x);
  CHECK_ERROR("VecDuplicate");
  ierr = VecCopy(v._x, _x);
  CHECK_ERROR("VecCopy");
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(PETScVector&& v) : _x(v._x) { v._x = nullptr; }
//-----------------------------------------------------------------------------
PETScVector::~PETScVector()
{
  if (_x)
    VecDestroy(&_x);
}
//-----------------------------------------------------------------------------
PETScVector& PETScVector::operator=(PETScVector&& v)
{
  if (_x)
    VecDestroy(&_x);
  _x = v._x;
  v._x = nullptr;
  return *this;
}
//-----------------------------------------------------------------------------
std::int64_t PETScVector::size() const
{
  if (!_x)
  {
    throw std::runtime_error(
        "PETSc vector has not been initialised. Cannot return size.");
  }

  PetscInt n = 0;
  PetscErrorCode ierr = VecGetSize(_x, &n);
  CHECK_ERROR("VecGetSize");
  return n;
}
//-----------------------------------------------------------------------------
std::size_t PETScVector::local_size() const
{
  if (!_x)
  {
    throw std::runtime_error(
        "PETSc vector has not been initialised. Cannot return local size.");
  }

  PetscInt n = 0;
  PetscErrorCode ierr = VecGetLocalSize(_x, &n);
  CHECK_ERROR("VecGetLocalSize");
  return n;
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> PETScVector::local_range() const
{
  if (!_x)
  {
    throw std::runtime_error(
        "PETSc vector has not been initialised. Cannot return local range.");
  }

  PetscInt n0, n1;
  PetscErrorCode ierr = VecGetOwnershipRange(_x, &n0, &n1);
  CHECK_ERROR("VecGetOwnershipRange");
  assert(n0 <= n1);
  return {{n0, n1}};
}
//-----------------------------------------------------------------------------
void PETScVector::get_local(std::vector<PetscScalar>& values) const
{
  assert(_x);
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
void PETScVector::set_local(const std::vector<PetscScalar>& values)
{
  assert(_x);
  const auto _local_range = local_range();
  const std::size_t local_size = _local_range[1] - _local_range[0];
  if (values.size() != local_size)
  {
    throw std::runtime_error("Cannot set local values of PETSc vector. Size of "
                             "values array is not equal to local vector size");
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
void PETScVector::add_local(const std::vector<PetscScalar>& values)
{
  assert(_x);
  const auto _local_range = local_range();
  const std::size_t local_size = _local_range[1] - _local_range[0];
  if (values.size() != local_size)
  {
    throw std::runtime_error("Cannot add local values to PETSc vector. Size of "
                             "values array is not equal to local vector size");
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
void PETScVector::get_local(PetscScalar* block, std::size_t m,
                            const PetscInt* rows) const
{
  if (m == 0)
    return;

  assert(_x);
  PetscErrorCode ierr;

  // Get ghost vector
  Vec xg = nullptr;
  ierr = VecGhostGetLocalForm(_x, &xg);
  CHECK_ERROR("VecGhostGetLocalForm");

  // Use array access if no ghost points, otherwise use VecGetValues on
  // local ghosted form of vector
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
    assert(xg);
    ierr = VecGetValues(xg, m, rows, block);
    CHECK_ERROR("VecGetValues");

    ierr = VecGhostRestoreLocalForm(_x, &xg);
    CHECK_ERROR("VecGhostRestoreLocalForm");
  }
}
//-----------------------------------------------------------------------------
void PETScVector::set(const PetscScalar* block, std::size_t m,
                      const PetscInt* rows)
{
  assert(_x);
  PetscErrorCode ierr = VecSetValues(_x, m, rows, block, INSERT_VALUES);
  CHECK_ERROR("VecSetValues");
}
//-----------------------------------------------------------------------------
void PETScVector::set_local(const PetscScalar* block, std::size_t m,
                            const PetscInt* rows)
{
  assert(_x);
  PetscErrorCode ierr = VecSetValuesLocal(_x, m, rows, block, INSERT_VALUES);
  CHECK_ERROR("VecSetValuesLocal");
}
//-----------------------------------------------------------------------------
void PETScVector::add(const PetscScalar* block, std::size_t m,
                      const PetscInt* rows)
{
  assert(_x);
  PetscErrorCode ierr = VecSetValues(_x, m, rows, block, ADD_VALUES);
  CHECK_ERROR("VecSetValues");
}
//-----------------------------------------------------------------------------
void PETScVector::add_local(const PetscScalar* block, std::size_t m,
                            const PetscInt* rows)
{
  assert(_x);
  PetscErrorCode ierr = VecSetValuesLocal(_x, m, rows, block, ADD_VALUES);
  CHECK_ERROR("VecSetValuesLocal");
}
//-----------------------------------------------------------------------------
void PETScVector::apply()
{
  common::Timer timer("Apply (PETScVector)");
  assert(_x);
  PetscErrorCode ierr;
  ierr = VecAssemblyBegin(_x);
  CHECK_ERROR("VecAssemblyBegin");
  ierr = VecAssemblyEnd(_x);
  CHECK_ERROR("VecAssemblyEnd");

  // Update any ghost values
  update_ghosts();
}
//-----------------------------------------------------------------------------
void PETScVector::apply_ghosts()
{
  assert(_x);
  PetscErrorCode ierr;

  Vec xg;
  ierr = VecGhostGetLocalForm(_x, &xg);
  CHECK_ERROR("VecGhostGetLocalForm");
  if (xg) // Vec is ghosted
  {
    ierr = VecGhostUpdateBegin(_x, ADD_VALUES, SCATTER_REVERSE);
    CHECK_ERROR("VecGhostUpdateBegin");
    ierr = VecGhostUpdateEnd(_x, ADD_VALUES, SCATTER_REVERSE);
    CHECK_ERROR("VecGhostUpdateEnd");
  }

  ierr = VecGhostRestoreLocalForm(_x, &xg);
  CHECK_ERROR("VecGhostRestoreLocalForm");
}
//-----------------------------------------------------------------------------
void PETScVector::update_ghosts()
{
  assert(_x);
  PetscErrorCode ierr;

  Vec xg;
  ierr = VecGhostGetLocalForm(_x, &xg);
  CHECK_ERROR("VecGhostGetLocalForm");
  if (xg) // Vec is ghosted
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
MPI_Comm PETScVector::mpi_comm() const
{
  assert(_x);
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  PetscErrorCode ierr = PetscObjectGetComm((PetscObject)(_x), &mpi_comm);
  CHECK_ERROR("PetscObjectGetComm");
  return mpi_comm;
}
//-----------------------------------------------------------------------------
void PETScVector::set(PetscScalar a)
{
  assert(_x);
  PetscErrorCode ierr = VecSet(_x, a);
  CHECK_ERROR("VecSet");
}
//-----------------------------------------------------------------------------
void PETScVector::shift(PetscScalar a)
{
  assert(_x);
  PetscErrorCode ierr = VecShift(_x, a);
  CHECK_ERROR("VecShift");
}
//-----------------------------------------------------------------------------
void PETScVector::scale(PetscScalar a)
{
  assert(_x);
  PetscErrorCode ierr = VecScale(_x, a);
  CHECK_ERROR("VecScale");
}
//-----------------------------------------------------------------------------
void PETScVector::mult(const PETScVector& v)
{
  assert(_x);
  assert(v._x);
  PetscErrorCode ierr = VecPointwiseMult(_x, _x, v._x);
  CHECK_ERROR("VecPointwiseMult");
}
//-----------------------------------------------------------------------------
bool PETScVector::empty() const { return _x == nullptr ? true : false; }
//-----------------------------------------------------------------------------
PetscScalar PETScVector::dot(const PETScVector& y) const
{
  assert(_x);
  assert(y._x);
  PetscScalar a;
  PetscErrorCode ierr = VecDot(_x, y._x, &a);
  CHECK_ERROR("VecDot");
  return a;
}
//-----------------------------------------------------------------------------
void PETScVector::axpy(PetscScalar a, const PETScVector& y)
{
  assert(_x);
  assert(y._x);
  PetscErrorCode ierr = VecAXPY(_x, a, y._x);
  CHECK_ERROR("VecAXPY");
}
//-----------------------------------------------------------------------------
void PETScVector::abs()
{
  assert(_x);
  PetscErrorCode ierr = VecAbs(_x);
  CHECK_ERROR("VecAbs");
}
//-----------------------------------------------------------------------------
PetscReal PETScVector::norm(la::Norm norm_type) const
{
  assert(_x);
  PetscErrorCode ierr;
  PetscReal value = 0.0;
  switch (norm_type)
  {
  case la::Norm::l1:
    ierr = VecNorm(_x, NORM_1, &value);
    CHECK_ERROR("VecNorm");
    break;
  case la::Norm::l2:
    ierr = VecNorm(_x, NORM_2, &value);
    CHECK_ERROR("VecNorm");
    break;
  case la::Norm::linf:
    ierr = VecNorm(_x, NORM_INFINITY, &value);
    CHECK_ERROR("VecNorm");
    break;
  default:
    throw std::runtime_error("Norm type not support for PETSc Vec");
  }

  return value;
}
//-----------------------------------------------------------------------------
PetscReal PETScVector::normalize()
{
  assert(_x);
  PetscReal norm;
  PetscErrorCode ierr = VecNormalize(_x, &norm);
  CHECK_ERROR("VecNormalize");
  return norm;
}
//-----------------------------------------------------------------------------
std::pair<double, PetscInt> PETScVector::min() const
{
  assert(_x);
  std::pair<double, PetscInt> data;
  PetscErrorCode ierr = VecMin(_x, &data.second, &data.first);
  CHECK_ERROR("VecMin");
  return data;
}
//-----------------------------------------------------------------------------
std::pair<double, PetscInt> PETScVector::max() const
{
  assert(_x);
  std::pair<double, PetscInt> data;
  PetscErrorCode ierr = VecMax(_x, &data.second, &data.first);
  CHECK_ERROR("VecMax");
  return data;
}
//-----------------------------------------------------------------------------
PetscScalar PETScVector::sum() const
{
  assert(_x);
  PetscScalar value = 0.0;
  PetscErrorCode ierr = VecSum(_x, &value);
  CHECK_ERROR("VecSum");
  return value;
}
//-----------------------------------------------------------------------------
std::string PETScVector::str(bool verbose) const
{
  assert(_x);
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
    assert(_x);
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
                         const std::vector<PetscInt>& indices) const
{
  assert(_x);
  PetscErrorCode ierr;

  // Get number of required entries
  const std::int64_t n = indices.size();

  // Check that passed vector is local
  if (MPI::size(y.mpi_comm()) != 1)
  {
    throw std::runtime_error(
        "PETSc gather vector must be a local vector (MPI_COMM_SELF).");
  }

  // Initialize vector if empty
  if (y.empty())
    y = PETScVector(PETSC_COMM_SELF, {{0, n}}, {}, 1);

  // Check that passed vector has correct size
  if (y.size() != n)
  {
    throw std::runtime_error(
        "PETSc gather vector must be empty or of correct size "
        "(same as provided indices)");
  }

  // Prepare data for index sets (global indices)
  std::vector<PetscInt> global_indices(indices.begin(), indices.end());

  // PETSc will bail out if it receives a NULL pointer even though m ==
  // 0.  Can't return from function since function calls are collective.
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
#if PETSC_VERSION_LE(3, 10, 100)
  ierr = VecScatterCreate(_x, from, y.vec(), to, &scatter);
#else
  ierr = VecScatterCreateWithData(_x, from, y.vec(), to, &scatter);
#endif
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
void PETScVector::set_options_prefix(std::string options_prefix)
{
  if (!_x)
  {
    throw std::runtime_error(
        "Cannot set options prefix. PETSc Vec has not been initialized.");
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
    throw std::runtime_error(
        "Cannot get options prefix. PETSc Vec has not been initialized.");
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
    throw std::runtime_error(
        "Cannot call VecSetFromOptions. PETSc Vec has not been initialized.");
  }

  PetscErrorCode ierr = VecSetFromOptions(_x);
  CHECK_ERROR("VecSetFromOptions");
}
//-----------------------------------------------------------------------------
Vec PETScVector::vec() const { return _x; }
//-----------------------------------------------------------------------------
