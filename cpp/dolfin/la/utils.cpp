// Copyright (C) 2013-2018 Johan Hake, Jan Blechta and Garth N. Wells
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

//-----------------------------------------------------------------------------
std::vector<IS> dolfin::la::compute_index_sets(
    std::vector<const dolfin::common::IndexMap*> maps)
{
  std::vector<IS> is(maps.size());
  std::size_t offset = 0;
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    assert(maps[i]);
    // if (MPI::rank(MPI_COMM_WORLD) == 1)
    //   std::cout << "CCC: " << i << ", " << maps[i]->size_local() << ", "
    //             << maps[i]->num_ghosts() << std::endl;
    const int size = maps[i]->size_local() + maps[i]->num_ghosts();
    std::vector<PetscInt> index(size);
    std::iota(index.begin(), index.end(), offset);
    ISCreateBlock(MPI_COMM_SELF, maps[i]->block_size(), index.size(),
                  index.data(), PETSC_COPY_VALUES, &is[i]);
    offset += size;
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
