// Copyright (C) 2008-2014 Anders Logg and Garth N. Wells
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
// This file provides DOLFIN typedefs for basic types.

#ifndef __DOLFIN_TYPES_H
#define __DOLFIN_TYPES_H

#ifdef HAS_PETSC
#include <petscsys.h>
#endif

namespace dolfin
{

  /// Index type for compatibility with linear algebra backend(s)
  #ifdef HAS_PETSC
  typedef PetscInt la_index;
  #else
  typedef int la_index;
  #endif

}

// Use la_index for indexing in Eigen::VectorXd
// NOTE: Eigen requires that this is SIGNED type
// NOTE: This file must be included before Eigen otherwise compiler should
//       complain about redefinition of this macro
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE dolfin::la_index

#endif
