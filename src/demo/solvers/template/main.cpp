// Copyright (C) 2002 [Insert name]
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

int main(int argc, char **argv)
{
  // Set problem
  dolfin_set_problem("my problem");
  
  // Set parameters
  dolfin_set_parameter("my parameter", 7.0);

  // Initialize dolfin
  dolfin_init(argc,argv);

  // Solve the problem
  dolfin_solve();
  
  // Finished
  dolfin_end();

  return 0;
}
