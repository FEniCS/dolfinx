// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

// Sparse matrix size
#define NS 50000

// Dense matrix size
#define ND 1000

int main()
{
  dolfin_set("output", "plain text");

  sysinfo();
  dolfin_info("");

  // Create sparse matrix and vectors
  Matrix AS(NS,NS);
  Vector xs(NS);
  Vector bs(NS);
  bs = 1.0;

  // Create dense matrix and vectors
  Matrix AD(ND,ND, Matrix::DENSE);
  Vector xd(ND);
  Vector bd(ND);
  bd = 1.0;
  
  // Assemble dense matrix (without timing)
  for (int i = 0; i < ND; i++) {
    AD(i,i) += 2.0;
    if ( i > 0 )
      AD(i, i - 1) += -1.0;
    if ( i < (ND-1) )
      AD(i, i + 1) += 1.0;
  }

  // Assembling
  dolfin_info("- Assembling 100 times");
  tic();
  for (int k = 0; k < 100; k++) {
    AS = 0.0;
    for (int i = 0; i < NS; i++) {
      AS(i,i) += 2.0;
      if ( i > 0 )
	AS(i, i - 1) += -1.0;
      if ( i < (NS-1) )
	AS(i, i + 1) += 1.0;
    }
  }
  real t1 = toc();
  
  // Matrix-vector multiplication
  dolfin_info("- Matrix/vector multiplication 1000 times");
  tic();
  for (int i = 0; i < 1000; i++)
    AS.mult(bs,xs);
  real t2 = toc();

  // LU factorization
  dolfin_info("- LU factorization");
  tic();
  AD.lu();
  real t3 = toc();

  // LU solve
  dolfin_info("- LU solve 100 times");
  tic();
  for (int i = 0; i < 100; i++)
    AD.solveLU(xd,bd);
  real t4 = toc();
  
  // Krylov solver  
  dolfin_info("- Krylov solve");
  tic();
  xs = 0.0;
  AS.solve(xs,bs);
  real t5 = toc();

  // Compute total time
  real t = t1 + t2 + t3 + t4 + t5;

  // Present results
  dolfin_info("");
  dolfin_info("Assembling:                   %.2f", t1);
  dolfin_info("Matrix/vector multiplication: %.2f", t2);
  dolfin_info("LU factorization:             %.2f", t3);
  dolfin_info("LU solve:                     %.2f", t4);
  dolfin_info("Krylov solve:                 %.2f", t5);
  dolfin_info("");
  dolfin_info("Total time:                   %.2f", t);

  dolfin_info("");
  dolfin_info("--------------------------------------------------------");
}
