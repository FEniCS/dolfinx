#include <iostream>
#include <mtl/mtl.h>
#include <dolfin/sysinfo.h>
#include <dolfin/meminfo.h>
#include <dolfin/timeinfo.h>

typedef double Real;

typedef mtl::matrix< Real, mtl::rectangle<>,
							mtl::array< mtl::compressed<> >,
							mtl::row_major>::type MTLSparseMatrix;

typedef mtl::dense1D<Real> MTLVector;

using namespace dolfin;

#define N 1000000

int main()
{
  cout << "Benchmark for linear algebra with MTL" << endl;
  cout << "-------------------------------------" << endl;
  sysinfo();
  
  cout << "Creating a vector with " << N << " elements" << endl;
  tic();
  MTLVector x(N);
  toc();
  meminfo();

  cout << "Creating another vector with " << N << " elements" << endl;
  tic();
  MTLVector y(N);
  toc();
  meminfo();
  
  cout << "Creating a " << N << " x " << N << " matrix" << endl;
  tic();
  MTLSparseMatrix A(N, N);
  toc();
  meminfo();

  cout << "Assembling" << endl;
  tic();
  for (int i = 0; i < N; i++) {
	 A(i,i) += 2.0;

	 if ( i > 0 )
		A(i, i - 1) += -1.0;

	 if ( i < (N-1) )
		A(i, i + 1) += 1.0;
  }
  toc();
  meminfo();

  cout << "Multiplying: y = A*x" << endl;
  tic();
  mult(A, x, y);
  toc();
  meminfo();
}
