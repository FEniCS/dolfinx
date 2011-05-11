// Copyright (C) 2011 Anders Logg and Garth N. Wells
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2011-02-11
// Last changed:

#ifdef HAS_PETSC

#include <dolfin/log/dolfin_log.h>
#include "GenericVector.h"
#include "PETScVector.h"
#include "PETScBaseMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::uint PETScBaseMatrix::size(uint dim) const
{
  assert(dim <= 1);
  if (A)
  {
    int m(0), n(0);
    MatGetSize(*A, &m, &n);
    if (dim == 0)
      return m;
    else
      return n;
  }
  else
    return 0;
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> PETScBaseMatrix::local_range(uint dim) const
{
  assert(dim <= 1);
  if (dim == 1)
    error("Cannot compute columns range for PETSc matrices.");
  if (A)
  {
    int m(0), n(0);
    MatGetOwnershipRange(*A, &m, &n);
    return std::make_pair(m, n);
  }
  else
    return std::make_pair(0, 0);
}
//-----------------------------------------------------------------------------
void PETScBaseMatrix::resize(GenericVector& y, uint dim) const
{
  assert(A);

  // Downcast vector
  PETScVector& _y = y.down_cast<PETScVector>();

  // Clear data
  _y.reset();

  // Create new PETSc vector
  boost::shared_ptr<Vec> x(new Vec(0), PETScVectorDeleter());
  if (dim == 0)
    MatGetVecs(*A, PETSC_NULL, x.get());
  else if (dim == 1)
    MatGetVecs(*A, x.get(), PETSC_NULL);
  else
    error("dim must be <= 1 when resizing vector.");

  // Associate new PETSc vector with _y
  _y.x = x;
}
//-----------------------------------------------------------------------------
#endif
