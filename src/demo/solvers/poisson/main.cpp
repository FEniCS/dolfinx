// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>
#include <math.h>

dolfin_bc my_bc (real x, real y, real z, int node, int component);
real      f     (real x, real y, real z, real t);

int main(int argc, char **argv)
{
  // Set problem
  dolfin_set_problem("poisson");
  
   // Set parameters
  dolfin_set_parameter("problem description", "Poisson's equation on the unit cube");
  dolfin_set_parameter("output file prefix",  "poisson");
  //dolfin_set_parameter("grid file",           "../../../data/grids/tetgrid_8_8_8.inp");
    
  // For 2d problems
  dolfin_set_parameter("grid file",       "../../../data/grids/trigrid.gid");
  dolfin_set_parameter("space dimension", 2);
  
  // Set boundary conditions
  dolfin_set_boundary_conditions(my_bc);
  
  // Set the right-hand side
  dolfin_set_function("source",f);
  
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

  if ( x == 0.0 ){
	 bc.type = dirichlet;
	 bc.val  = 0.0;
  }
  if ( x == 1.0 ){
	 bc.type = dirichlet;
	 bc.val  = 0.0;
  }

  return bc;
}

real f(real x, real y, real z, real t)
{
  real dx = x - 0.5;
  real dy = y - 0.5;
  real r  = sqrt( dx*dx + dy*dy );

  if ( r < 0.3 )
	 return 100.0;
  else
	 return 0.0;
}
