// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-10-21
// Last changed: 2005-12-12

#include <stdlib.h>
#include <dolfin.h>

using namespace dolfin;

int main(int argc, char** argv)
{
  dolfin_output("plain text");

  if ( argc != 3 ) {
    dolfin::cout << "Usage: dolfin-ode method q' where method is one of" << dolfin::endl;
    dolfin::cout << "cg or dg, and q is the order" << dolfin::endl;
    return 1;
  }
  
  int q = atoi(argv[2]);

  if ( strcmp(argv[1], "cg") == 0 ) {
    
    cGqMethod cGq(q);
    cGq.disp();

  }
  else if ( strcmp(argv[1], "dg") == 0 ) {
    
    dGqMethod dGq(q);
    dGq.disp();
    
  } 
  else {
    dolfin::cout << "Unknown method." << dolfin::endl;
    return 1;
  }
  
  return 0;
}
