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

  cout << b << endl;
  cout << A << endl;

  cout << x << endl;

  /*
  A(1,1) = 0.5;
  real a = A(1,1);
  cout << "a = " << a << endl;
  */

  cout << "b = " << b << endl;

  cout << "b = "; b.show();
  cout << "A = "; A.show();

  cout << "x = "; x.show();

  KrylovSolver ks;

  //  DenseMatrix D(A);
  //  D.DisplayAll();

  ks.solve(A,x,b);

  cout << "x = "; x.show();

  Vector R;

  R.init(x.size());
  A.mult(x,R);
  b *= -1.0;
  R += b;

  cout << "R = "; R.show();
  

}
