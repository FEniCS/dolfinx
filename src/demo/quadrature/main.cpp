// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdlib.h>
#include <dolfin.h>

using namespace dolfin;

int main(int argc, char** argv)
{
  dolfin_set("output", "plain text");

  //if ( argc != 3 ) {
  //  dolfin::cout << "Usage: dolfin-quadrature rule n' where rule is one of" << dolfin::endl;
  //  dolfin::cout << "gauss, radau, lobatto, and n is the number of points" << dolfin::endl;
  //  return 1;
  // }

  int n = atoi(argv[2]);

  n = 3;

  //  if ( strcasecmp(argv[1], "gauss") == 0 ) {
    
    GaussQuadrature q(n);
    dolfin::cout << q << dolfin:: endl;

    /*}
  else if ( strcasecmp(argv[1], "radau") == 0 ) {

    RadauQuadrature q(n);

  }
  else if ( strcasecmp(argv[1], "lobatto") == 0 ) {

    LobattoQuadrature q(n);

  }
  else {
    dolfin::cout << "Unknown quadrature rule." << dolfin::endl;
    return 1;
    }*/

  return 0;
}
