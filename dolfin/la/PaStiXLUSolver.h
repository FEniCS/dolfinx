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
// First added:  2011-10-16
// Last changed:

#ifndef __DOLFIN_PASTIXLUSOLVER_H
#define __DOLFIN_PASTIXLUSOLVER_H

#include <utility>
#include <vector>
#include <petscconf.h>
#include <dolfin/common/types.h>

#ifdef PETSC_HAVE_PASTIX

namespace dolfin
{

  // Forward declarations
  class GenericMatrix;
  class GenericVector;
  class SparsityPattern;
  class STLMatrix;

  class PaStiXLUSolver
  {
  public:

    /// Constructor
    PaStiXLUSolver(const SparsityPattern& pattern, const STLMatrix& A, GenericVector& x, const GenericVector& b);

    /// Destructor
    virtual ~PaStiXLUSolver();

    /// Solve linear system Ax = b
    unsigned int solve(GenericVector& x, const GenericVector& b);

    unsigned int size(unsigned int dim) const
    { return _size[dim]; }

  private:

    // Gobal size
    unsigned int _size[2];

    const uint id;
    std::pair<unsigned int, unsigned int> row_range;

  };

}

#endif
#endif
