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

  cout << "b = " << b << endl;

  cout << "b = "; b.show();
  cout << "A = "; A.show();

  cout << "x = "; x.show();

  KrylovSolver ks;

  ks.solve(A,x,b);

  cout << "x = "; x.show();

  Vector R;

  R.init(x.size());
  A.mult(x,R);
  b *= -1.0;
  R += b;

  cout << "R = "; R.show();
}
