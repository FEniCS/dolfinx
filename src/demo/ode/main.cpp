// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdlib.h>
#include <dolfin.h>

using namespace dolfin;

int main(int argc, char** argv)
{
  dolfin_output("plain text");

  if ( argc != 3 ) {
    dolfin::cout << "Usage: dolfin-ode method q' where method is one of" << dolfin::endl;
    dolfin::cout << "cgq or dgq, and q is the order" << dolfin::endl;
    return 1;
  }
  
  int q = atoi(argv[2]);

  if ( strcasecmp(argv[1], "cgq") == 0 ) {
    
    NewcGqMethod cGq(q);
    cGq.disp();

  }
  else if ( strcasecmp(argv[1], "dgq") == 0 ) {
    
    NewdGqMethod dGq(q);
    dGq.disp();
    
  } 
  else {
    dolfin::cout << "Unknown method." << dolfin::endl;
    return 1;
  }
  
  return 0;
}
