// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Andy R. Terrel, 2005.
//
// First added:  2005-01-17
// Last changed: 2009-08-10

#ifdef HAS_PETSC

#include <iostream>
#include <petscmat.h>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/log/dolfin_log.h>
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
    boost::shared_ptr<Vec> _x(&x, NoDeleter<Vec>());
    boost::shared_ptr<Vec> _y(&y, NoDeleter<Vec>());

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
PETScKrylovMatrix::PETScKrylovMatrix(const PETScVector& x, const PETScVector& y)
{
  // Create PETSc matrix
  init(x, y);
}
//-----------------------------------------------------------------------------
PETScKrylovMatrix::~PETScKrylovMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PETScKrylovMatrix::init(const PETScVector& x, const PETScVector& y)
{
  // Get size and local size of given vector
  int m(0), n(0), M(0), N(0);
  VecGetLocalSize(*(y.vec()), &m);
  VecGetLocalSize(*(x.vec()), &n);
  VecGetSize(*(y.vec()), &M);
  VecGetSize(*(x.vec()), &N);

  if (A)
  {
    // Get size and local size of existing matrix
    int mm(0), nn(0), MM(0), NN(0);
    MatGetLocalSize(*A, &mm, &nn);
    MatGetSize(*A, &MM, &NN);

    if ( mm == m && nn == n && MM == M && NN == N )
      return;
    else
     A.reset(new Mat, PETScMatrixDeleter());
  }
  else
    A.reset(new Mat, PETScMatrixDeleter());

  MatCreateShell(PETSC_COMM_WORLD, m, n, M, N, (void*) this, A.get());
  MatShellSetOperation(*A, MATOP_MULT, (void (*)()) usermult);
}
//-----------------------------------------------------------------------------
void PETScKrylovMatrix::resize(int M, int N)
{
  // Put here to set up arbitrary Shell of global size M x N
  // Analagous to the matrix being on one processor.

  if (A)
  {
    // Get size and local size of existing matrix
    int MM(0), NN(0);
    MatGetSize(*A, &MM, &NN);

    if ( MM == M && NN == N )
      return;
    else
      A.reset(new Mat, PETScMatrixDeleter());
  }
  else
    A.reset(new Mat, PETScMatrixDeleter());

  MatCreateShell(PETSC_COMM_WORLD, M, N, M, N, (void*) this, A.get());
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
