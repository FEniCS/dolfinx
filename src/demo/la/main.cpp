// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

int main(int argc, char **argv)
{  
  Vector x;
  Matrix A;
  
  File file("data.xml");
  
  file >> x;
  file >> A;

  cout << x << endl;
  cout << A << endl;

  A(1,1) = 0.5;
  real a = A(1,1);
  cout << "a = " << a << endl;
  
  x.show();
  A.show();
}
