// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <math.h>
#include <petscksp.h>
#include <petscerror.h>
#include <dolfin.h>
#include <dolfin/GMRES.h>

using namespace dolfin;

class IdentityPreconditioner : public Preconditioner
{
public:
  virtual ~IdentityPreconditioner()
  {
  };

  virtual void solve(Vector& x, const Vector& b)
  {
    cout << "preconditioning (identity)" << endl;

    // Do nothing

    x = b;

  };
};

class JacobiPreconditioner : public Preconditioner
{
public:
  JacobiPreconditioner(Matrix &A) : A(A)
  {
  };
  virtual ~JacobiPreconditioner()
  {
  };

  virtual void solve(Vector& x, const Vector& b)
  {
    cout << "preconditioning (jacobi)" << endl;

    // Diagonal preconditioner

    // Note: inefficient operators, don't use these when you need efficiency

    int n = x.size();

    for(int i = 0; i < n; i++)
    {
      x(i) = b(i) / A(i, i);
    }
  };

  Matrix &A;
};


int main(int argc, char **argv)
{  
  //petsc(argc, argv);

  //dolfin_set("output", "plain text");

  dolfin::cout << "--------------------------------------" << dolfin::endl;
  dolfin::cout << "Test new LA matrix notation" << dolfin::endl;
  dolfin::cout << "--------------------------------------" << dolfin::endl;
  
  Vector c(10);
  dolfin::cout << "c: " << dolfin::endl;
  c.disp();

  c(3) = 12.3;
  c(7) = 45.6;

  dolfin::cout << "c: " << dolfin::endl;
  c.disp();

  real r;
  r = c(7);

  dolfin::cout << "c(7): " << r << dolfin::endl;


  Matrix B(4, 4);
  B.apply();

  B(2, 3) = 1.234;

  real b11 = B(1, 1);
  real b23 = B(2, 3);

  dolfin::cout << "B: " << dolfin::endl;
  B.disp();

  dolfin::cout << "B(1, 1): " << b11 << dolfin::endl;
  dolfin::cout << "B(2, 3): " << b23 << dolfin::endl;


  dolfin::cout << "--------------------------------------" << dolfin::endl;
  dolfin::cout << "Test file i/o" << dolfin::endl;
  dolfin::cout << "--------------------------------------" << dolfin::endl;

  Vector bold, xold;
  Matrix Aold;

  File file("data.xml");

  file >> bold;
  file >> Aold;
  
  dolfin::cout << "bold = "; 
  bold.show();

  dolfin::cout << "Aold = "; 
  Aold.show();

  Vector b(bold), x(b.size());
  Matrix A(Aold);

  dolfin::cout << "b: " << dolfin::endl;
  b.disp();

  dolfin::cout << "A: " << dolfin::endl;
  A.disp(false);
  
  dolfin::cout << "--------------------------------------" << dolfin::endl;
  dolfin::cout << "Test Krylov solvers" << dolfin::endl;
  dolfin::cout << "--------------------------------------" << dolfin::endl;


  KrylovSolver ks;
  Vector Rold;


  dolfin_set("krylov tolerance",1.0e-10);
  dolfin_set("max no krylov restarts", 100);
  dolfin_set("max no stored krylov vectors", 100);
  dolfin_set("max no cg iterations", 100);
  dolfin_set("pc iterations", 5);

  xold = 1.0;
  ks.setMethod(KrylovSolver::GMRES);
  ks.solve(Aold,xold,bold);

  Rold.init(xold.size());
  Aold.mult(xold, Rold);
  Rold -= bold;

  dolfin::cout << "xold = ";
  xold.show();
  dolfin::cout << "Rold = ";
  Rold.show();



  GMRES newsolver;
  Vector R(x.size());


  newsolver.solve(A, x, b);
  A.mult(x, R);
  R.axpy(-1, b);

  dolfin::cout << "x: " << dolfin::endl;
  x.disp();

  dolfin::cout << "R: " << dolfin::endl;
  R.disp();


  dolfin::cout << "--------------------------------------" << dolfin::endl;
  dolfin::cout << "Test Preconditioner interface" << dolfin::endl;
  dolfin::cout << "--------------------------------------" << dolfin::endl;

  const int N = 61;

  Matrix A2(N, N);
  Vector b2(N), x2(N), R2(N);

  for(int i = 0; i < N; i++)
  {
    A2(i, i) = 2.0;
    b2(i) = 1.0 / N;
    x2(i) = 1.0;
  }

  for(int i = 0; i < N - 1; i++)
  {
    A2(i + 1, i) = -1;
    A2(i, i + 1) = -1;
  }

//   cout << "A2: " << endl;
//   A2.disp(false);
//   cout << "b2: " << endl;
//   b2.disp();
//   cout << "x2: " << endl;
//   x2.disp();



  GMRES newsolver2;

  IdentityPreconditioner idpc;
  //JacobiPreconditioner jpc(A2);
  newsolver2.setPreconditioner(idpc);
  //newsolver2.setPreconditioner(jpc);

  x2 = 1.0;
  newsolver2.solve(A2, x2, b2);

  A2.mult(x2, R2);
  R2.axpy(-1, b2);

  dolfin::cout << "x2: " << dolfin::endl;
  x2.disp();

  dolfin::cout << "R2: " << dolfin::endl;
  R2.disp();

}


