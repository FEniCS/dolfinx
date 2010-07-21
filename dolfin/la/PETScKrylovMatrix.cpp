// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Andy R. Terrel, 2005.
//
// First added:  2005-01-17
// Last changed: 2009-08-10

#ifdef HAS_PETSC

#include <boost/shared_ptr.hpp>
#include <iostream>

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
PETScKrylovMatrix::PETScKrylovMatrix(): A(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScKrylovMatrix::PETScKrylovMatrix(const PETScVector& x, const PETScVector& y)
  : A(0)
{
  // Create PETSc matrix
  init(x, y);
}
//-----------------------------------------------------------------------------
PETScKrylovMatrix::~PETScKrylovMatrix()
{
  // Free memory of matrix
  if ( A )
    MatDestroy(A);
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

  // Free previously allocated memory if necessary
  if ( A )
  {
    // Get size and local size of existing matrix
    int mm(0), nn(0), MM(0), NN(0);
    MatGetLocalSize(A, &mm, &nn);
    MatGetSize(A, &MM, &NN);

    if ( mm == m && nn == n && MM == M && NN == N )
      return;
    else
      MatDestroy(A);
  }

  MatCreateShell(PETSC_COMM_WORLD, m, n, M, N, (void*) this, &A);
  MatShellSetOperation(A, MATOP_MULT, (void (*)()) usermult);
}
//-----------------------------------------------------------------------------
void PETScKrylovMatrix::resize(int M, int N)
{
  // Put here to set up arbitrary Shell of global size M,N.
  // Analagous to the matrix being on one processor.

  // Free previously allocated memory if necessary
  if ( A )
  {
    // Get size and local size of existing matrix
    int MM(0), NN(0);
    MatGetSize(A, &MM, &NN);

    if ( MM == M && NN == N )
      return;
    else
      MatDestroy(A);
  }

  MatCreateShell(PETSC_COMM_WORLD, M, N, M, N, (void*) this, &A);
  MatShellSetOperation(A, MATOP_MULT, (void (*)()) usermult);
}
//-----------------------------------------------------------------------------
dolfin::uint PETScKrylovMatrix::size(uint dim) const
{
  int M = 0;
  int N = 0;
  MatGetSize(A, &M, &N);
  assert(M >= 0);
  assert(N >= 0);

  return (dim == 0 ? static_cast<uint>(M) : static_cast<uint>(N));
}
//-----------------------------------------------------------------------------
Mat PETScKrylovMatrix::mat() const
{
  return A;
}
//-----------------------------------------------------------------------------
std::string PETScKrylovMatrix::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    warning("Verbose output for PETScKrylovMatrix not implemented.");
  }
  else
  {
    s << "<PETScKrylovMatrix of size " << size(0) << " x " << size(1) << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------

#endif
