// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Vector.h>
#include <dolfin/SparseMatrix.h>
#include <dolfin/sysinfo.h>
#include <dolfin/meminfo.h>
#include <dolfin/timeinfo.h>

using namespace dolfin;

#define N 1000000

int main()
{
  cout << "Benchmark for linear algebra in DOLFIN" << endl;
  cout << "--------------------------------------" << endl;
  sysinfo();
  
  cout << "Creating a vector with " << N << " elements" << endl;
  tic();
  Vector x(N);
  toc();
  meminfo();

  cout << "Creating another vector with " << N << " elements" << endl;
  tic();
  Vector y(N);
  toc();
  meminfo();
  
  cout << "Creating a " << N << " x " << N << " matrix" << endl;
  tic();
  SparseMatrix A(N, N);
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
  A.mult(x, y);
  toc();
  meminfo();
}
