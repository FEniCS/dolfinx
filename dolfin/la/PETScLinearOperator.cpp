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
// Modified by Andy R. Terrel 2005
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
#include "PETScLinearOperator.h"

using namespace dolfin;

// Callback function for PETSc mult function
namespace dolfin
{
  int usermult(Mat A, Vec x, Vec y)
  {
    // Wrap PETSc Vec as dolfin::PETScVector
    boost::shared_ptr<Vec> _x(&x, NoDeleter());
    boost::shared_ptr<Vec> _y(&y, NoDeleter());
    PETScVector __x(_x);
    PETScVector __y(_y);

    // Extract pointer to PETScLinearOperator
    void* ctx = 0;
    MatShellGetContext(A, &ctx);
    PETScLinearOperator* _A = ((PETScLinearOperator*) ctx);

    // Call user-defined mult function through wrapper
    dolfin_assert(_A);
    GenericLinearOperator* wrapper = _A->wrapper();
    dolfin_assert(wrapper);
    wrapper->mult(__x, __y);

    return 0;
  }
}

//-----------------------------------------------------------------------------
PETScLinearOperator::PETScLinearOperator() : _wrapper(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t PETScLinearOperator::size(std::size_t dim) const
{
  return PETScBaseMatrix::size(dim);
}
//-----------------------------------------------------------------------------
void PETScLinearOperator::mult(const GenericVector& x, GenericVector& y) const
{
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
std::string PETScLinearOperator::str(bool verbose) const
{
  std::stringstream s;
  if (verbose)
  {
    warning("Verbose output for PETScLinearOperator not implemented.");
    s << str(false);
  }
  else
  {
    s << "<PETScLinearOperator of size "
      << PETScBaseMatrix::size(0)
      << " x "
      << PETScBaseMatrix::size(1) << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
const GenericLinearOperator* PETScLinearOperator::wrapper() const
{
  return _wrapper;
}
//-----------------------------------------------------------------------------
GenericLinearOperator* PETScLinearOperator::wrapper()
{
  return _wrapper;
}
//-----------------------------------------------------------------------------
void PETScLinearOperator::init(std::size_t M, std::size_t N, GenericLinearOperator* wrapper)
{
  // Store wrapper
  _wrapper = wrapper;

  // Compute local range
  const std::pair<std::size_t, std::size_t> row_range    = MPI::local_range(M);
  const std::pair<std::size_t, std::size_t> column_range = MPI::local_range(N);
  const std::size_t m_local = row_range.second - row_range.first;
  const std::size_t n_local = column_range.second - column_range.first;

  // Check whether matrix has already been initialized and dimensions match
  if (A)
  {
    // Get size and local size of existing matrix
    PetscInt _M(0), _N(0), _m_local(0), _n_local(0);
    MatGetSize(*A, &_M, &_N);
    MatGetLocalSize(*A, &_m_local, &_n_local);

    // Check whether size already matches
    if (M == static_cast<std::size_t>(_M) &&
        N == static_cast<std::size_t>(_N) &&
        m_local == static_cast<std::size_t>(_m_local) &&
        n_local == static_cast<std::size_t>(_n_local))
    {
      return;
    }
  }

  // Initialize PETSc matrix
  A.reset(new Mat, PETScMatrixDeleter());
  MatCreateShell(PETSC_COMM_WORLD, m_local, n_local, M, N, (void*) this, A.get());
  MatShellSetOperation(*A, MATOP_MULT, (void (*)()) usermult);
}
//-----------------------------------------------------------------------------

#endif
