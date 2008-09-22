// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-08-18
// Last changed:
//
// Benchmarks for uBLAS matrices

#include <dolfin.h>
#include <boost/tuple/tuple.hpp>
#include "Poisson.h"
#include "VectorPoisson.h"

using namespace dolfin;
using namespace boost::tuples;

//-----------------------------------------------------------------------------
template<class Mat, class Vec = uBLASVector>
struct MatrixAssemble
{
  //---------------------------------------------------------------------------
  static real assemble(Form& a, const dolfin::uint N)
  {
    dolfin_set("output destination", "silent");  
    UnitSquare mesh(N, N);
    Mat A;
  
    tic();
    dolfin::assemble(A, a, mesh);
    return toc();
  }
  //---------------------------------------------------------------------------
  static real assemble(Form& a, Mat& A, const dolfin::uint N)
  {
    dolfin_set("output destination", "silent");  
    UnitSquare mesh(N, N);

    tic();
    Assembler::assemble(A, a, mesh);
    return toc();
  }
  //---------------------------------------------------------------------------
  static tuple<real, real> assemble(dolfin::uint N)
  {
    dolfin_set("output destination", "silent");
    tuple<real, real> timing;
    Mat A;
    A.init(2*N, 2*N);
    A.zero();
    
    real block[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; 
    dolfin::uint ipos[4] = {0, 0, 0, 0};
    dolfin::uint jpos[4] = {0, 0, 0, 0};

    tic();
    for(dolfin::uint i = 0; i < N-1; ++i)
    {
      ipos[0] = i;   ipos[1] = i+1; ipos[2] = N + i; ipos[3] = N+i+1;
      jpos[0] = i+1; jpos[1] = i;   jpos[2] = N+i+1; jpos[3] = N+i;
      A.add(block, 4, ipos, 4, jpos);
    }
//    get<0>(timing) = toc();  
    timing.get<0>() = toc();  
  
    tic();
    A.apply();
    get<1>(timing) = toc();  

    return timing;    
  }
  //---------------------------------------------------------------------------
  static real vector_multiply(const dolfin::uint N, const dolfin::uint n)
  {
    dolfin_set("output destination", "silent");
    VectorPoissonBilinearForm a;
    Mat A;
    Vec x, y;
    UnitSquare mesh(N,N);  

    dolfin::assemble(A, a, mesh); 
    x.init(A.size(1));
    y.init(A.size(1));

    tic();
    for(dolfin::uint i = 0; i < n; ++i)
      A.mult(x, y); 
    return toc();
  }
};
//-----------------------------------------------------------------------------
void AssemblePoissonMatrix()
{
  // Assembly of sparse matrices on a N x N unit mesh
  real time;
  const dolfin::uint n = 3;
  const dolfin::uint N[n] = {5, 10, 40};
  PoissonBilinearForm a;
  VectorPoissonBilinearForm a_vector;

  begin("Assemble a sparse matrix for scalar Poisson equation on an square N x N mesh");
#ifdef HAS_PETSC  
  for(dolfin::uint i =0; i < n; ++i)
  {
    time = MatrixAssemble< PETScMatrix >::assemble(a, N[i]);
    dolfin_set("output destination", "terminal");
    cout << "PETScMatrix       (N=" << N[i] << "): " << time << endl;
  }
#endif
  for(dolfin::uint i =0; i < n; ++i)
  {
    time = MatrixAssemble< uBLASMatrix<ublas_sparse_matrix> >::assemble(a, N[i]);
    dolfin_set("output destination", "terminal");
    cout << "uBLASSparseMatrix (N=" << N[i] << "): " << time << endl;
  }
  end();

  begin("Assemble a sparse matrix for vector Poisson equation on an square N x N mesh");
#ifdef HAS_PETSC  
  for(dolfin::uint i =0; i < n; ++i)
  {
    time = MatrixAssemble< PETScMatrix >::assemble(a_vector, N[i]);
    dolfin_set("output destination", "terminal");
    cout << "PETScMatrix       (N=" << N[i] << "): " << time << endl;
  }
#endif
  for(dolfin::uint i =0; i < n; ++i)
  {
    time = MatrixAssemble< uBLASMatrix<ublas_sparse_matrix> >::assemble(a_vector, N[i]);
    dolfin_set("output destination", "terminal");
    cout << "uBLASSparseMatrix (N=" << N[i] << "): " << time << endl;
  }
  end();
}
//-----------------------------------------------------------------------------
void AssembleSparseMatrices()
{
  // Assemble of sparse matrices of size 2*N x 2*N
  const dolfin::uint N = 5000;
  tuple<real, real> timing;

  dolfin_set("output destination", "terminal");
  begin("Assemble a sparse matrix in quasi-random order (size = 2N x 2N)" );
#ifdef HAS_PETSC  
  timing = MatrixAssemble< PETScMatrix >::assemble(N);
  dolfin_set("output destination", "terminal");
  cout << "PETScMatrix insert         (N=" << N << "): " << get<0>(timing) << endl;
  cout << "PETScMatrix finalise       (N=" << N << "): " << get<1>(timing) << endl;
#endif
  timing = MatrixAssemble< uBLASMatrix<ublas_sparse_matrix> >::assemble(N);
  dolfin_set("output destination", "terminal");
  cout << "uBLASSparseMatrix insert   (N=" << N << "): " << get<0>(timing) << endl;
  cout << "uBLASSparseMatrix finalise (N=" << N << "): " << get<01>(timing) << endl;
        
  end();
}
//-----------------------------------------------------------------------------
void MatrixVectorMultiply()
{
  // Assembly of sparse matrices on a N x N unit mesh
  const dolfin::uint n = 20;
  const dolfin::uint N = 20;
  real time;

  dolfin_set("output destination", "terminal");
  begin("Sparse matrix-vector multiplication (size N x N, repeated n times)");
#ifdef HAS_PETSC 
  time = MatrixAssemble< PETScMatrix, PETScVector >::vector_multiply(N, n);
  dolfin_set("output destination", "terminal");
  cout << "PETScMatrix: (N=" << N << ", n=" << n << "): " << time << endl;
#endif
  time = MatrixAssemble< uBLASMatrix<ublas_sparse_matrix>, uBLASVector >::vector_multiply(N, n);
  dolfin_set("output destination", "terminal");
  cout << "uBLASMatrix: (N=" << N << ", n=" << n << "): " << time << endl;

  end();
}
//-----------------------------------------------------------------------------
int main()
{
  cout << "Initialisation of a sparse matrix needs updating for this benchmark." << endl;
  return 0;  

  begin("Sparse matrix benchmark timings");
  dolfin_set("output destination", "silent");

  // Assembly of a sparse matrix
  AssembleSparseMatrices();  

  // Assembly of a Poisson problem (this should really be in FEM)
  AssemblePoissonMatrix();  

  // Sparse matrix - vector multiplication
  MatrixVectorMultiply();  

  return 0;
}
