// Copyright (C) 2013-2019 Johan Hake, Jan Blechta and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <cassert>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/SubSystemsManager.h>
#include <dolfin/log/log.h>
#include <petsc.h>

#include <dolfin/fem/Form.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/function/FunctionSpace.h>

//-----------------------------------------------------------------------------
std::vector<std::vector<const dolfin::common::IndexMap*>>
dolfin::la::bocked_index_sets(
    const std::vector<std::vector<const dolfin::fem::Form*>> a)
{
  std::vector<std::vector<const common::IndexMap*>> maps(2);
  maps[0].resize(a.size());
  maps[1].resize(a[0].size());

  // Loop over rows and columns
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    for (std::size_t j = 0; j < a[i].size(); ++j)
    {
      if (a[i][j])
      {
        assert(a[i][j].rank() == 2);
        auto m0 = a[i][j]->function_space(0)->dofmap()->index_map();
        auto m1 = a[i][j]->function_space(1)->dofmap()->index_map();
        if (!maps[0][i])
          maps[0][i] = m0.get();
        else
        {
          // TODO: Check that maps are the same
        }

        if (!maps[1][j])
          maps[1][j] = m1.get();
        else
        {
          // TODO: Check that maps are the same
        }
      }
    }
  }

  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    for (std::size_t j = 0; j < maps[i].size(); ++j)
    {
      if (!maps[i][j])
        throw std::runtime_error("Could not deduce all block index maps.");
    }
  }

  return maps;
}
//-----------------------------------------------------------------------------
std::vector<IS> dolfin::la::compute_index_sets(
    std::vector<const dolfin::common::IndexMap*> maps)
{
  std::vector<IS> is(maps.size());
  std::size_t offset = 0;
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    assert(maps[i]);
    const int size = maps[i]->size_local() + maps[i]->num_ghosts();
    const int bs = maps[i]->block_size();
    std::vector<PetscInt> index(bs * size);
    std::iota(index.begin(), index.end(), offset);
    ISCreateBlock(MPI_COMM_SELF, 1, index.size(), index.data(),
                  PETSC_COPY_VALUES, &is[i]);
    // ISCreateBlock(MPI_COMM_SELF, bs, index.size(), index.data(),
    //               PETSC_COPY_VALUES, &is[i]);
    offset += bs * size;
    // offset += size;
  }

  return is;
}
//-----------------------------------------------------------------------------
void dolfin::la::petsc_error(int error_code, std::string filename,
                             std::string petsc_function)
{
  // Fetch PETSc error description
  const char* desc;
  PetscErrorMessage(error_code, &desc, nullptr);

  // Fetch and clear PETSc error message
  const std::string msg = common::SubSystemsManager::singleton().petsc_err_msg;
  dolfin::common::SubSystemsManager::singleton().petsc_err_msg = "";

  // Log detailed error info
  dolfin::log::log(TRACE, "PETSc error in '%s', '%s'", filename.c_str(),
                   petsc_function.c_str());
  dolfin::log::log(
      TRACE, "PETSc error code '%d' (%s), message follows:", error_code, desc);
  // NOTE: don't put msg as variadic argument; it might get trimmed
  dolfin::log::log(TRACE, std::string(78, '-'));
  dolfin::log::log(TRACE, msg);
  dolfin::log::log(TRACE, std::string(78, '-'));

  // Raise exception with standard error message
  throw std::runtime_error("Failed to successfully call PETSc function '"
                           + petsc_function + "'. PETSc error code is: "
                           + std ::to_string(error_code) + ", "
                           + std::string(desc));
}
//-----------------------------------------------------------------------------
dolfin::la::VecWrapper::VecWrapper(Vec y, bool ghosted) : x(nullptr, 0), _y(y)
{
  assert(_y);
  if (ghosted)
    VecGhostGetLocalForm(_y, &_y_local);
  else
    VecGetLocalVector(_y, _y_local);

  PetscInt n = 0;
  VecGetSize(_y_local, &n);
  VecGetArray(_y_local, &array);

  new (&x) Eigen::Map<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>(array, n);
}
//-----------------------------------------------------------------------------
void dolfin::la::VecWrapper::restore()
{
  assert(_y_local);
  VecRestoreArray(_y_local, &array);

  PetscBool is_ghost_local_form;
  assert(_y);
  VecGhostIsLocalForm(_y, _y_local, &is_ghost_local_form);
  if (is_ghost_local_form == PETSC_TRUE)
    VecGhostRestoreLocalForm(_y, &_y_local);
  else
    VecRestoreLocalVector(_y, _y_local);
}
//-----------------------------------------------------------------------------
dolfin::la::VecReadWrapper::VecReadWrapper(const Vec y, bool ghosted)
    : x(nullptr, 0), _y(y)
{
  assert(_y);
  if (ghosted)
    VecGhostGetLocalForm(_y, &_y_local);
  else
    VecGetLocalVector(_y, _y_local);

  PetscInt n = 0;
  VecGetSize(_y_local, &n);
  VecGetArrayRead(_y_local, &array);
  new (&x)
      Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>(array, n);
}
//-----------------------------------------------------------------------------
void dolfin::la::VecReadWrapper::restore()
{
  assert(_y_local);
  VecRestoreArrayRead(_y_local, &array);

  assert(_y);
  PetscBool is_ghost_local_form;
  VecGhostIsLocalForm(_y, _y_local, &is_ghost_local_form);
  if (is_ghost_local_form == PETSC_TRUE)
    VecGhostRestoreLocalForm(_y, &_y_local);
  else
    VecRestoreLocalVector(_y, _y_local);
}
//-----------------------------------------------------------------------------
