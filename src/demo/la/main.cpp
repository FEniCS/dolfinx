// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

int main(int argc, char **argv)
{  
  Vector x;
  Vector b;
  Matrix A;
  
  File file("data.xml");
  
  file >> b;
  file >> A;

  cout << A << endl;
  cout << b << endl;
  cout << x << endl;

  //  A(1,1) = 0.5;
  real a = A(1,1);
  cout << "A(1,1) = " << a << endl;
  cout << "A(2,1) = " << A(2,1) << endl;

  cout << "A = "; A.show();
  cout << "b = "; b.show();
  cout << "x = "; x.show();

  KrylovSolver ks;

  x = 1.0;
  ks.setMethod(KrylovSolver::gmres);
  ks.solve(A,x,b);

  x = 1.0;
  ks.setMethod(KrylovSolver::cg);
  ks.solve(A,x,b);

  SISolver si;

  x = 1.0;
  si.setMethod(SISolver::jacobi);
  si.solve(A,x,b);

  x = 1.0;
  si.setMethod(SISolver::gauss_seidel);
  si.solve(A,x,b);

  x = 1.0;
  si.setMethod(SISolver::sor);
  si.solve(A,x,b);

  x = 1.0;
  si.setMethod(SISolver::richardson);
  si.solve(A,x,b);

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
  si.setMethod(SISolver::jacobi);
  si.solve(A,x,b);

  x = 1.0;
  si.setMethod(SISolver::gauss_seidel);
  si.solve(A,x,b);

  x = 1.0;
  si.setMethod(SISolver::sor);
  si.solve(A,x,b);

  x = 1.0;
  si.setMethod(SISolver::richardson);
  si.solve(A,x,b);

  Vector R;
  R.init(x.size());
  A.mult(x,R);
  b *= -1.0;
  R += b;

  cout << "x = "; x.show();
  cout << "R = "; R.show();
}
