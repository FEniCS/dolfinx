// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

int main(int argc, char **argv)
{  
  dolfin_set("output", "plain text");

  dolfin::cout << "--------------------------------------" << dolfin::endl;
  dolfin::cout << "Test new LA matrix notation" << dolfin::endl;
  dolfin::cout << "--------------------------------------" << dolfin::endl;
  
  Vector c(10);
  c.show();

  c(3) = 12.3;
  c(7) = 45.6;

  c.show();

  real r = c(7);

  dolfin::cout << "c(7): " << r << dolfin::endl;

  /*
  dolfin::cout << "--------------------------------------" << dolfin::endl;
  dolfin::cout << "Test file i/o" << dolfin::endl;
  dolfin::cout << "--------------------------------------" << dolfin::endl;
  
  Vector x;
  Vector b;
  Matrix A;
  
  File file("data.xml");
  File matlab("matrix.m");
  
  file >> b;
  file >> A;

  dolfin::cout << A << dolfin::endl;
  dolfin::cout << b << dolfin::endl;
  dolfin::cout << x << dolfin::endl;

  matlab << A;

  dolfin::cout << "A = "; 
  A.show();
  dolfin::cout << "b = "; 
  b.show();
  dolfin::cout << "x = "; 
  x.show();
  */


  /*

  dolfin::cout << "--------------------------------------" << dolfin::endl;
  dolfin::cout << "Test matrix notation" << dolfin::endl;
  dolfin::cout << "--------------------------------------" << dolfin::endl;
  
  Matrix B(A);
  B.show();

  Vector c(b);
  c.show();
 
  B(3,all) = c;
  B.show();

  B(all,5) = c; 
  B.show();

  c = B(all,first);
  c.show();

  B(last,all) = B(all,2);
  B.show();

  dolfin::cout << "--------------------------------------" << dolfin::endl;
  dolfin::cout << "Test Krylov solvers" << dolfin::endl;
  dolfin::cout << "--------------------------------------" << dolfin::endl;

  KrylovSolver ks;
  Vector R;


  dolfin_set("krylov tolerance",1.0e-10);
  dolfin_set("max no krylov restarts", 100);
  dolfin_set("max no stored krylov vectors", 100);
  dolfin_set("max no cg iterations", 100);
  dolfin_set("pc iterations", 5);

  x = 1.0;
  ks.setMethod(KrylovSolver::GMRES);
  ks.solve(A,x,b);
  R.init(x.size()); A.mult(x,R); R -= b;
  dolfin::cout << "x = "; x.show(); dolfin::cout << "R = "; R.show();

  x = 1.0;
  ks.setMethod(KrylovSolver::CG);
  ks.solve(A,x,b);
  R.init(x.size()); A.mult(x,R); R -= b;
  dolfin::cout << "x = "; x.show(); dolfin::cout << "R = "; R.show();

  dolfin::cout << "--------------------------------------" << dolfin::endl;
  dolfin::cout << "Test for stationary iterative solvers" << dolfin::endl;
  dolfin::cout << "--------------------------------------" << dolfin::endl;

  SISolver si;

  x = 1.0;
  si.setMethod(SISolver::JACOBI);
  si.solve(A,x,b);
  R.init(x.size()); A.mult(x,R); R -= b;
  dolfin::cout << "x = "; x.show(); dolfin::cout << "R = "; R.show();

  x = 1.0;
  si.setMethod(SISolver::GAUSS_SEIDEL);
  si.solve(A,x,b);
  R.init(x.size()); A.mult(x,R); R -= b;
  dolfin::cout << "x = "; x.show(); dolfin::cout << "R = "; R.show();

  x = 1.0;
  si.setMethod(SISolver::SOR);
  si.solve(A,x,b);
  R.init(x.size()); A.mult(x,R); R -= b;
  dolfin::cout << "x = "; x.show(); dolfin::cout << "R = "; R.show();

  x = 1.0;
  si.setMethod(SISolver::RICHARDSON);
  si.solve(A,x,b);
  R.init(x.size()); A.mult(x,R); R -= b;
  dolfin::cout << "x = "; x.show(); dolfin::cout << "R = "; R.show();

  // Change to diagonal dominant matrix (should be simple for SI methods) 
  for (int i=0; i < 10; i++){
    A(i,i) = 1.0;
    if (i==0){
      A(0,1) = 1.0e-1;
    } else if (i==9){
      A(9,8) = 1.0e-1;
    } else{ 
      A(i,i-1) = 1.0e-1;
      A(i,i+1) = 1.0e-1;
    }
  }
  dolfin::cout << "A = "; A.show();

  x = 1.0;
  si.setMethod(SISolver::JACOBI);
  si.solve(A,x,b);
  R.init(x.size()); A.mult(x,R); R -= b;
  dolfin::cout << "x = "; x.show(); dolfin::cout << "R = "; R.show();

  x = 1.0;
  si.setMethod(SISolver::GAUSS_SEIDEL);
  si.solve(A,x,b);
  R.init(x.size()); A.mult(x,R); R -= b;
  dolfin::cout << "x = "; x.show(); dolfin::cout << "R = "; R.show();

  x = 1.0;
  si.setMethod(SISolver::SOR);
  si.solve(A,x,b);
  R.init(x.size()); A.mult(x,R); R -= b;
  dolfin::cout << "x = "; x.show(); dolfin::cout << "R = "; R.show();

  x = 1.0;
  si.setMethod(SISolver::RICHARDSON);
  si.solve(A,x,b);
  R.init(x.size()); A.mult(x,R); R -= b;
  dolfin::cout << "x = "; x.show(); dolfin::cout << "R = "; R.show();

  dolfin::cout << "--------------------------------------" << dolfin::endl;
  dolfin::cout << "Benchmark for Krylov solvers in DOLFIN" << dolfin::endl;
  dolfin::cout << "--------------------------------------" << dolfin::endl;

  int N = 200;

  A.init(N,N);
  b.init(N);

  Vector U1(N);
  Vector U2(N);
  Vector U3(N);
  Vector U4(N);

  dolfin::cout << "Assembling stiffness matrix" << dolfin::endl;
  tic();
  for (int i = 0; i < N; i++) {
    A(i,i) = 20.0;
    if ( i > 0 )
      A(i, i - 1) = -10.0;
    if ( i < (N-1) )
      A(i, i + 1) = -10.0;
  }
  toc();
  meminfo();
  
  dolfin::cout << "Computing load vector" << dolfin::endl;
  tic();
  for (int i = 0; i < N; i++) {
    b(i) = 10.0;
  }
  toc();
  meminfo();

  if(N <= 10)
  {
    A.show();
    b.show();
  }


  dolfin::cout << "Solving Ax = b using CG" << dolfin::endl;
  tic();
  ks.setMethod(KrylovSolver::CG);
  ks.solve(A, U1, b);
  toc();
  meminfo();
  R.init(U1.size()); A.mult(U1,R); R -= b;
  if(N <= 10){ dolfin::cout << "x = "; U1.show(); dolfin::cout << "R = "; R.show();}

  dolfin::cout << "Solving Ax = b using GMRES" << dolfin::endl;
  tic();
  ks.setMethod(KrylovSolver::GMRES);
  ks.solve(A, U2, b);
  toc();
  meminfo();
  R.init(U2.size()); A.mult(U2,R); R -= b;
  if(N <= 10){ dolfin::cout << "x = "; U2.show(); dolfin::cout << "R = "; R.show();}

  // Preconditioning (simple diagonal scaling)
  real d;
  for (int i = 0; i < N; i++) {
    d = A(i,i);
    b(i)   = b(i)/d;
    for (int j = 0; j < N; j++) {
      A(i,j) = A(i,j)/d;
    }
  }
  
  if(N <= 10)
  {
    A.show();
    b.show();
  }

  dolfin::cout << "Solving Ax = b using CG, preconditioned" << dolfin::endl;
  tic();
  ks.setMethod(KrylovSolver::CG);
  ks.solve(A, U3, b);
  toc();
  meminfo();
  R.init(U3.size()); A.mult(U3,R); R -= b;
  if(N <= 10){ dolfin::cout << "x = "; U3.show(); dolfin::cout << "R = "; R.show();}

  dolfin::cout << "Solving Ax = b using GMRES, preconditioned" << dolfin::endl;
  tic();
  ks.setMethod(KrylovSolver::GMRES);
  ks.solve(A, U4, b);
  toc();
  meminfo();
  R.init(U4.size()); A.mult(U4,R); R -= b;
  if(N <= 10){ dolfin::cout << "x = "; U4.show(); dolfin::cout << "R = "; R.show();}

  real diff_norm = 0.0;
  for (int i = 0; i < N; i++) diff_norm += pow(U1(i)-U3(i),2.0);
  diff_norm = sqrt(diff_norm);
  dolfin::cout << "Difference wrt preconditioned CG solutions: " << diff_norm << dolfin::endl;

  diff_norm = 0.0;
  for (int i = 0; i < N; i++) diff_norm += pow(U2(i)-U4(i),2.0);
  diff_norm = sqrt(diff_norm);
  dolfin::cout << "Difference wrt preconditioned GMRES solutions: " << diff_norm << dolfin::endl;

  */

}


