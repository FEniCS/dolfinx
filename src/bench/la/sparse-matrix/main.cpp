// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-08-18
// Last changed:
//
// Benchmarks for uBlas matrices

#include <dolfin.h>
#include <boost/tuple/tuple.hpp>
#include "Poisson.h"
#include "VectorPoisson.h"

using namespace dolfin;
using namespace boost::tuples;

//-----------------------------------------------------------------------------
template<class Mat>
struct MatrixAssemble
{
  static tuple<real, real> benchUBlasSparseAssemble(const dolfin::uint N)
  {
    dolfin_log(false);

    tuple<real, real> timing;

    UnitSquare mesh(N,N);

    Poisson::BilinearForm a;
    VectorPoisson::BilinearForm a_vector;
    Mat A, A_vector;

    tic();
    FEM::assemble(a, A, mesh);
    get<0>(timing) = toc();

    tic();
    FEM::assemble(a_vector, A_vector, mesh);
    get<1>(timing) = toc();

    dolfin_log(true);

    return timing;
  }
};
//-----------------------------------------------------------------------------
int main()
{
  // Assembly of sparse matrices on a N x N unit mesh
  const dolfin::uint N = 500;

  tuple<real, real> timing;
  timing = MatrixAssemble< uBlasMatrix<ublas_sparse_matrix> >::benchUBlasSparseAssemble(N);
  cout << "Time to assemble uBlas sparse matrix for scalar Poisson equation: " << get<0>(timing) << endl;
  cout << "Time to assemble uBlas sparse matrix for vector Poisson equation: " << get<1>(timing) << endl;

#ifdef HAVE_PETSC_H  
  timing = MatrixAssemble< PETScMatrix >::benchUBlasSparseAssemble(N);
  cout << "Time to assemble PETSc sparse matrix for scalar Poisson equation: " << get<0>(timing) << endl;
  cout << "Time to assemble PETSc sparse matrix for vector Poisson equation: " << get<1>(timing) << endl;
#endif


 
  return 0;
}
