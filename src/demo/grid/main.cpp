// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

int main()
{
  dolfin_set("create edges",true); 

  Grid grid;
  File file1("grid.xml.gz");
  File file2("grid_refined.dx");

  // Read grid from file
  file1 >> grid;
  dolfin::cout << grid << dolfin::endl;  
  
  // Show the entire grid
  //grid.show();

  // Refine grid
  grid.refine();
  
  // Save refined grid to file
  file2 << grid;

  dolfin::cout << grid << dolfin::endl;  

  return 0;
}
