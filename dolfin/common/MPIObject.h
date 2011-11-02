// Copyright (C) 2011 Garth N. Wells
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
// First added:  2011-11-02
// Last changed:

#ifndef __DOLFIN_MPI_OBJECT_H
#define __DOLFIN_MPI_OBJECT_H

#include <dolfin/common/SubSystemsManager.h>

namespace dolfin
{
  /// This class initialises MPI is not already initialised.

  class MPIObject
  {
  public:

    MPIObject() { SubSystemsManager::init_mpi(); }

  };

}

#endif
