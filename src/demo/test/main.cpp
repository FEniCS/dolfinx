// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

int main()
{
  dolfin::cout << "Test" << " test" << dolfin::endl;

  Matrix A(3,3, Matrix::DENSE);

  A(0,0) = 1.0;
  A(0,2) = 2.0;
  A(1,0) = 3.0;
  A(2,1) = 1.0;
  A(2,2) = 2.0;

  Vector b(3);
  b = 1.0;

  Vector x;
  A.solve(x,b);

  x.show();

  return 0;
}
