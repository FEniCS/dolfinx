// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

int main()
{
  dolfin::cout << "Test" << " test" << dolfin::endl;

  Lagrange p(0);
  p.set(0, 0.0);
  dolfin::cout << "p(0.2) = " << p(0, 0.2) << dolfin::endl;

  return 0;
}
