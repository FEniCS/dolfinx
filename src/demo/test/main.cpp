// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

int main()
{
  dolfin_set("output", "plain text");

  dolfin::cout << "Test" << " test" << dolfin::endl;
  
  dolfin_load("dolfin.xml");
  dolfin_save("dolfin2.xml");

  return 0;
}
