// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

int main()
{
  dolfin::cout << "Test" << " test" << dolfin::endl;

  Matrix A(3,3);

  real a = A(0,0);
  
  return 0;
}
