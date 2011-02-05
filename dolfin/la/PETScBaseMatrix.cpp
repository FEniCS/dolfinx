// Copyright (C) 20011 Anders Logg and Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
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
    MatGetVecs(*A, x.get(), PETSC_NULL);
  else if (dim == 1)
    MatGetVecs(*A, PETSC_NULL, x.get());
  else
    error("dim must be <= 1 when resizing vector.");

  // Associate new PETSc vector with _y
  _y.x = x;
}
//-----------------------------------------------------------------------------
#endif
