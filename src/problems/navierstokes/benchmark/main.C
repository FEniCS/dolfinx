// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

dolfin_bc my_bc(real x, real y, real z, int node, int component);

real fx(real x, real y, real z, real t);

real Re = 100.0;

int main(int argc, char **argv)
{
  // Set problem
  dolfin_set_problem("navier-stokes");
  
  // Set parameters
  dolfin_set_parameter("problem description", "Navier-Stokes equations");
  dolfin_set_parameter("grid file",           "../../../../data/grids/tetgrid_8_8_8.inp");
  dolfin_set_parameter("output file prefix",  "navier_stokes");
  dolfin_set_parameter("output file type",  "opendx");
  dolfin_set_parameter("output samples",10);
  dolfin_set_parameter("reynolds number", Re);
  dolfin_set_parameter("start time", 0.0);
  dolfin_set_parameter("final time", 0.1);
  dolfin_set_parameter("debug level", 0);

  dolfin_set_function("fx",fx);

  // Set boundary conditions
  dolfin_set_boundary_conditions(my_bc);
  
  // Initialize dolfin
  dolfin_init(argc,argv);

  // Save parameters
  dolfin_save_parameters();

  // Solve the problem
  dolfin_solve();
  
  // Finished
  dolfin_end();
  
  return 0;
}

dolfin_bc my_bc(real x, real y, real z, int node, int component)
{
  dolfin_bc bc;

  switch ( component ){
  case 0:
    
    /*
    if ( x == 0.0 ){
      bc.type = dirichlet;
      bc.val  = 1.0;
    }
    if ( x == 1.0 ){
      bc.type = dirichlet;
      bc.val  = 1.0;
    }
    */
    if ( y == 0.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( y == 1.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( z == 0.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( z == 1.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }

    break;
  case 1:

    /*
    if ( x == 0.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( x == 1.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    */
    if ( y == 0.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( y == 1.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( z == 0.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( z == 1.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }

    break;

  case 2:

    /* 
    if ( x == 0.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( x == 1.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    */
    if ( y == 0.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( y == 1.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( z == 0.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( z == 1.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    
    break;

  case 3:

    if ( x == 0.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( x == 1.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }

    break;

  }
    
  return bc;
}

real fx(real x, real y, real z, real t)
{
  return (32.0/Re) * (y*(1.0-y) + z*(1.0-z)); 
}
