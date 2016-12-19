// Copyright (C) 2016 Patrick E. Farrell and Garth N. Wells
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

#ifndef __PETSC_DM_COLLECTION_H
#define __PETSC_DM_COLLECTION_H

#ifdef HAS_PETSC

#include <memory>
#include <vector>
#include <petscdm.h>
#include <petscvec.h>
#include <dolfin/la/PETScObject.h>
#include <dolfin/log/log.h>

namespace dolfin
{

  class FunctionSpace;

  class PETScDMCollection : public PETScObject
  {
  public:

    /// Constructor
    PETScDMCollection(std::vector<std::shared_ptr<const FunctionSpace>> function_spaces);

    /// Destructor
    ~PETScDMCollection();

    DM fine()
    {
      dolfin_assert(!_dms.empty());
      return _dms.back();
    }


  private:

    static PetscErrorCode create_global_vector(DM dm, Vec* vec);
    static PetscErrorCode create_interpolation(DM dmc, DM dmf, Mat *mat, Vec *vec);
    static PetscErrorCode coarsen(DM dmf, MPI_Comm comm, DM* dmc);
    static PetscErrorCode refine(DM dmc, MPI_Comm comm, DM* dmf);

    std::vector<DM> _dms;

  };

}

#endif

#endif
