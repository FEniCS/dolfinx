// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace std;
using namespace dolfin;

int main(int argc, char **argv)
{  
  cout << "--------------------------------------" << endl;
  cout << "Test for i/o on file" << endl;
  cout << "--------------------------------------" << endl;
  
  Vector x;
  Vector b;
  Matrix A;
  
  File file("data.xml");
  File matlab("matrix.m");
  
  file >> b;
  file >> A;

  cout << A << endl;
  cout << b << endl;
  cout << x << endl;

  matlab << A;
  
  real a = A(1,1);
  cout << "A(1,1) = " << a << endl;
  cout << "A(2,1) = " << A(2,1) << endl;

  cout << "A = "; 
  A.show();
  cout << "b = "; 
  b.show();
  cout << "x = "; 
  x.show();

  cout << "--------------------------------------" << endl;
  cout << "Test for Krylov solvers" << endl;
  cout << "--------------------------------------" << endl;

  KrylovSolver ks;
  Vector R;

  x = 1.0;
  ks.setMethod(KrylovSolver::GMRES);
  ks.solve(A,x,b);
  R.init(x.size()); A.mult(x,R); R -= b;
  cout << "x = "; x.show(); cout << "R = "; R.show();

  x = 1.0;
  ks.setMethod(KrylovSolver::CG);
  ks.solve(A,x,b);
  R.init(x.size()); A.mult(x,R); R -= b;
  cout << "x = "; x.show(); cout << "R = "; R.show();

  cout << "--------------------------------------" << endl;
  cout << "Test for stationary iterative solvers" << endl;
  cout << "--------------------------------------" << endl;

  SISolver si;

  x = 1.0;
  si.setMethod(SISolver::JACOBI);
  si.solve(A,x,b);
  R.init(x.size()); A.mult(x,R); R -= b;
  cout << "x = "; x.show(); cout << "R = "; R.show();

  x = 1.0;
  si.setMethod(SISolver::GAUSS_SEIDEL);
  si.solve(A,x,b);
  R.init(x.size()); A.mult(x,R); R -= b;
  cout << "x = "; x.show(); cout << "R = "; R.show();

  x = 1.0;
  si.setMethod(SISolver::SOR);
  si.solve(A,x,b);
  R.init(x.size()); A.mult(x,R); R -= b;
  cout << "x = "; x.show(); cout << "R = "; R.show();

  x = 1.0;
  si.setMethod(SISolver::RICHARDSON);
  si.solve(A,x,b);
  R.init(x.size()); A.mult(x,R); R -= b;
  cout << "x = "; x.show(); cout << "R = "; R.show();

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
  cout << "A = "; A.show();

  x = 1.0;
  si.setMethod(SISolver::JACOBI);
  si.solve(A,x,b);
  R.init(x.size()); A.mult(x,R); R -= b;
  cout << "x = "; x.show(); cout << "R = "; R.show();

  x = 1.0;
  si.setMethod(SISolver::GAUSS_SEIDEL);
  si.solve(A,x,b);
  R.init(x.size()); A.mult(x,R); R -= b;
  cout << "x = "; x.show(); cout << "R = "; R.show();

  x = 1.0;
  si.setMethod(SISolver::SOR);
  si.solve(A,x,b);
  R.init(x.size()); A.mult(x,R); R -= b;
  cout << "x = "; x.show(); cout << "R = "; R.show();

  x = 1.0;
  si.setMethod(SISolver::RICHARDSON);
  si.solve(A,x,b);
  R.init(x.size()); A.mult(x,R); R -= b;
  cout << "x = "; x.show(); cout << "R = "; R.show();

  cout << "--------------------------------------" << endl;
  cout << "Benchmark for Krylov solvers in DOLFIN" << endl;
  cout << "--------------------------------------" << endl;

  int N = 200;

  A.init(N,N);
  b.init(N);

  Vector U1(N);
  Vector U2(N);
  Vector U3(N);
  Vector U4(N);

  cout << "Assembling stiffness matrix" << endl;
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
  
  cout << "Computing load vector" << endl;
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


  cout << "Solving Ax = b using CG" << endl;
  tic();
  ks.setMethod(KrylovSolver::CG);
  ks.solve(A, U1, b);
  toc();
  meminfo();
  R.init(U1.size()); A.mult(U1,R); R -= b;
  if(N <= 10){ cout << "x = "; U1.show(); cout << "R = "; R.show();}

  cout << "Solving Ax = b using GMRES" << endl;
  tic();
  ks.setMethod(KrylovSolver::GMRES);
  ks.solve(A, U2, b);
  toc();
  meminfo();
  R.init(U2.size()); A.mult(U2,R); R -= b;
  if(N <= 10){ cout << "x = "; U2.show(); cout << "R = "; R.show();}

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

  cout << "Solving Ax = b using CG, preconditioned" << endl;
  tic();
  ks.setMethod(KrylovSolver::CG);
  ks.solve(A, U3, b);
  toc();
  meminfo();
  R.init(U3.size()); A.mult(U3,R); R -= b;
  if(N <= 10){ cout << "x = "; U3.show(); cout << "R = "; R.show();}

  cout << "Solving Ax = b using GMRES, preconditioned" << endl;
  tic();
  ks.setMethod(KrylovSolver::GMRES);
  ks.solve(A, U4, b);
  toc();
  meminfo();
  R.init(U4.size()); A.mult(U4,R); R -= b;
  if(N <= 10){ cout << "x = "; U4.show(); cout << "R = "; R.show();}

  real diff_norm = 0.0;
  for (int i = 0; i < N; i++) diff_norm += pow(U1(i)-U3(i),2.0);
  diff_norm = sqrt(diff_norm);
  cout << "Difference wrt preconditioned CG solutions: " << diff_norm << endl;

  diff_norm = 0.0;
  for (int i = 0; i < N; i++) diff_norm += pow(U2(i)-U4(i),2.0);
  diff_norm = sqrt(diff_norm);
  cout << "Difference wrt preconditioned GMRES solutions: " << diff_norm << endl;

}


