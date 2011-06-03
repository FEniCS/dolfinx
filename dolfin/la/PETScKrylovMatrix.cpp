// Copyright (C) 2005-2006 Anders Logg and Garth N. Wells
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
// Modified by Andy R. Terrel, 2005.
//
// First added:  2005-01-17
// Last changed: 2010-08-28

#ifdef HAS_PETSC

#include <iostream>
#include <petscmat.h>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/types.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/MPI.h>
#include "PETScVector.h"
#include "PETScKrylovMatrix.h"

using namespace dolfin;

// Mult function

// FIXME: Add an explanation of how this this function works
namespace dolfin
{
  int usermult(Mat A, Vec x, Vec y)
  {
    // Wrap x and y in a shared_ptr
    boost::shared_ptr<Vec> _x(&x, NoDeleter());
    boost::shared_ptr<Vec> _y(&y, NoDeleter());

    void* ctx = 0;
    MatShellGetContext(A, &ctx);
    PETScVector xx(_x), yy(_y);
    ((PETScKrylovMatrix*) ctx)->mult(xx, yy);
    return 0;
  }
}
//-----------------------------------------------------------------------------
PETScKrylovMatrix::PETScKrylovMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScKrylovMatrix::PETScKrylovMatrix(dolfin::uint m, dolfin::uint n)
{
  resize(m, n);
}
//-----------------------------------------------------------------------------
PETScKrylovMatrix::~PETScKrylovMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PETScKrylovMatrix::resize(dolfin::uint m, dolfin::uint n)
{
  // Compute local range
  const std::pair<uint, uint> row_range    = MPI::local_range(m);
  const std::pair<uint, uint> column_range = MPI::local_range(n);
  const int m_local = row_range.second - row_range.first;
  const int n_local = column_range.second - column_range.first;

  if (A)
  {
    // Get size and local size of existing matrix
    int _m(0), _n(0), _m_local(0), _n_local(0);
    MatGetSize(*A, &_m, &_m);
    MatGetLocalSize(*A, &_m_local, &_n_local);

    if (static_cast<int>(m) == _m && static_cast<int>(n) == _n && m_local == _m_local && n_local == _n_local )
      return;
    else
      A.reset(new Mat, PETScMatrixDeleter());
  }
  else
    A.reset(new Mat, PETScMatrixDeleter());

  MatCreateShell(PETSC_COMM_WORLD, m_local, n_local, m, n, (void*) this, A.get());
  MatShellSetOperation(*A, MATOP_MULT, (void (*)()) usermult);
}
//-----------------------------------------------------------------------------
std::string PETScKrylovMatrix::str(bool verbose) const
{
  std::stringstream s;
  if (verbose)
    warning("Verbose output for PETScKrylovMatrix not implemented.");
  else
    s << "<PETScKrylovMatrix of size " << size(0) << " x " << size(1) << ">";

  return s.str();
}
//-----------------------------------------------------------------------------

#endif
